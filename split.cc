#include <string>
#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cassert>

#include "io.hh"
#include "defs.hh"
#include "conf.hh"
#include "ExchangeAlgorithm.hh"

using namespace std;


struct SplitEvalTask {
    SplitEvalTask() : cidx(-1), ll(-DBL_MAX) { }
    int cidx;
    double ll;
    set<int> class1_words;
    set<int> class2_words;
};


void write_super_classes(string scfname,
                         vector<set<int> > &super_classes,
                         map<int, int> &super_class_lookup)
{
    cerr << "Writing super class definitions to " << scfname << endl;

    SimpleFileOutput classf(scfname);

    for (int i=0; i<(int)super_classes.size(); i++) {
        for (auto cit=super_classes[i].begin(); cit != super_classes[i].end(); ++cit)
        {
            if (cit != super_classes[i].begin()) classf << ",";
            classf << *cit;
        }
        classf << "\n";
    }
}



void find_candidate_classes(Exchange &e,
                            vector<int> &classes_to_evaluate,
                            set<int> stoplist,
                            int num_classes)
{
    double m_num_word_types=(double)e.m_vocabulary.size();
    double m_num_word_tokens=0.0;
    for (int i=0; i<(int)e.m_word_counts.size(); i++)
        m_num_word_tokens += (double)e.m_word_counts[i];

    multimap<double, int> class_order;
    for (int i=2; i<(int)e.m_classes.size(); i++) {
        if (e.m_classes[i].size() < 2) continue;
        if (stoplist.find(i) != stoplist.end()) continue;
        double score = 0.5 * (double)e.m_classes.size() / m_num_word_types;
        score += 0.5 * (double)e.m_class_counts[i] / m_num_word_tokens;
        class_order.insert(make_pair(score, i));
    }

    auto coit = class_order.rbegin();
    classes_to_evaluate.clear();
    while ((int)classes_to_evaluate.size() < num_classes) {
        classes_to_evaluate.push_back(coit->second);
        coit++;
    }
}


void split_classes(Exchange &e,
                   int target_num_classes,
                   int num_split_evals,
                   double ll_threshold,
                   string model_fname,
                   int model_write_interval,
                   vector<set<int> > &super_classes,
                   map<int, int> &super_class_lookup)
{
    srand(0);

    set<int> stoplist;

    while (e.num_classes() < target_num_classes)
    {
        vector<int> classes_to_evaluate;
        find_candidate_classes(e, classes_to_evaluate, stoplist, 50);
        int class1_idx = classes_to_evaluate[0];

        SplitEvalTask best_split;
        best_split.cidx = class1_idx;

        if (num_split_evals == 0) {
            cerr << "random split for class " << best_split.cidx << ", size: " << e.m_classes[best_split.cidx].size() << endl;
            e.random_split(e.m_classes[best_split.cidx], best_split.class1_words, best_split.class2_words);
        }
        else {
            cerr << "evaluating " << num_split_evals << " random splits for class "
                    << best_split.cidx << ", size: " << e.m_classes[best_split.cidx].size() << endl;
            for (int i=0; i<num_split_evals; i++) {
                SplitEvalTask split_task;
                split_task.cidx = class1_idx;
                e.random_split(e.m_classes[split_task.cidx], split_task.class1_words, split_task.class2_words);
                double orig_ll = e.log_likelihood();
                int class2_idx = e.do_split(split_task.cidx, split_task.class1_words, split_task.class2_words);
                e.iterate_exchange_local_2(split_task.cidx, class2_idx);
                double split_ll = e.log_likelihood();
                split_task.ll = split_ll - orig_ll;
                if (split_task.ll > best_split.ll)
                    best_split = split_task;
                e.do_merge(split_task.cidx, class2_idx);
            }
        }

        if (best_split.ll < ll_threshold) {
            stoplist.insert(class1_idx);
            continue;
        }

        cerr << "splitting.." << endl;
        int class2_idx = e.do_split(best_split.cidx, best_split.class1_words, best_split.class2_words);
        cerr << e.num_classes() << "\t" << e.log_likelihood() << endl;
        cerr << "running local exchange algorithm.." << endl;
        e.iterate_exchange_local_2(best_split.cidx, class2_idx);
        cerr << "final class sizes: " << e.m_classes[best_split.cidx].size() << " "
                << e.m_classes[class2_idx].size() << endl;
        cerr << e.num_classes() << "\t" << e.log_likelihood() << endl;

        int sci = super_class_lookup[best_split.cidx];
        super_classes[sci].insert(class2_idx);
        super_class_lookup[class2_idx] = sci;

        if (model_write_interval > 0 && e.num_classes() % model_write_interval == 0) {
            e.write_word_classes(model_fname + "." + int2str(e.num_classes()) + ".cgenprobs.gz");
            e.write_class_mem_probs(model_fname + "." + int2str(e.num_classes()) + ".cmemprobs.gz");
            e.write_classes(model_fname + "." + int2str(e.num_classes()) + ".classes.gz");
        }
    }
}


void split_classes_2(Exchange &e,
                     int target_num_classes,
                     int num_split_evals,
                     double ll_threshold,
                     string model_fname,
                     int model_write_interval,
                     vector<set<int> > &super_classes,
                     map<int, int> &super_class_lookup)
{
    srand(0);

    set<int> stoplist;

    while (e.num_classes() < target_num_classes)
    {
        vector<int> classes_to_evaluate;
        find_candidate_classes(e, classes_to_evaluate, stoplist, 50);

        SplitEvalTask best_split;

        if (num_split_evals == 0) {
            cerr << "random split for class " << best_split.cidx << ", size: " << e.m_classes[best_split.cidx].size() << endl;
            e.random_split(e.m_classes[best_split.cidx], best_split.class1_words, best_split.class2_words);
        }
        else {
            int num_evaluated_classes = 5;
            for (int c=0; c<num_evaluated_classes; c++) {
                SplitEvalTask curr_best;
                int curr_num_split_evals = (int)round(double(num_split_evals)/double(num_evaluated_classes));
                for (int i=0; i<curr_num_split_evals; i++) {
                    SplitEvalTask split_task;
                    split_task.cidx = classes_to_evaluate[c];
                    e.random_split(e.m_classes[split_task.cidx], split_task.class1_words, split_task.class2_words);
                    double orig_ll = e.log_likelihood();
                    int class2_idx = e.do_split(split_task.cidx, split_task.class1_words, split_task.class2_words);
                    e.iterate_exchange_local_2(split_task.cidx, class2_idx, 2);
                    double split_ll = e.log_likelihood();
                    split_task.ll = split_ll - orig_ll;
                    if (split_task.ll > best_split.ll)
                        best_split = split_task;
                    if (split_task.ll > curr_best.ll)
                        curr_best = split_task;
                    e.do_merge(split_task.cidx, class2_idx);
                }
                if (curr_best.ll < ll_threshold)
                    stoplist.insert(curr_best.cidx);
            }

            for (int i=0; i<num_split_evals; i++) {
                SplitEvalTask split_task;
                split_task.cidx = best_split.cidx;
                e.random_split(e.m_classes[split_task.cidx], split_task.class1_words, split_task.class2_words);
                double orig_ll = e.log_likelihood();
                int class2_idx = e.do_split(split_task.cidx, split_task.class1_words, split_task.class2_words);
                e.iterate_exchange_local_2(split_task.cidx, class2_idx, 2);
                double split_ll = e.log_likelihood();
                split_task.ll = split_ll - orig_ll;
                if (split_task.ll > best_split.ll)
                    best_split = split_task;
                e.do_merge(split_task.cidx, class2_idx);
            }
        }

        cerr << "splitting.." << endl;
        int class2_idx = e.do_split(best_split.cidx, best_split.class1_words, best_split.class2_words);
        cerr << e.num_classes() << "\t" << e.log_likelihood() << endl;
        cerr << "running local exchange algorithm.." << endl;
        e.iterate_exchange_local_2(best_split.cidx, class2_idx);
        cerr << "final class sizes: " << e.m_classes[best_split.cidx].size() << " "
                << e.m_classes[class2_idx].size() << endl;
        cerr << e.num_classes() << "\t" << e.log_likelihood() << endl;

        int sci = super_class_lookup[best_split.cidx];
        super_classes[sci].insert(class2_idx);
        super_class_lookup[class2_idx] = sci;

        if (model_write_interval > 0 && e.num_classes() % model_write_interval == 0) {
            write_super_classes(model_fname + "." + int2str(e.num_classes()) + ".superclasses.gz",
                                super_classes,
                                super_class_lookup);
            e.write_word_classes(model_fname + "." + int2str(e.num_classes()) + ".cgenprobs.gz");
            e.write_class_mem_probs(model_fname + "." + int2str(e.num_classes()) + ".cmemprobs.gz");
            e.write_classes(model_fname + "." + int2str(e.num_classes()) + ".classes.gz");
        }
    }
}


int main(int argc, char* argv[])
{
    try {
        conf::Config config;
        config("usage: split [OPTION...] CORPUS CLASS_INIT MODEL\n")
        ('c', "num-classes=INT", "arg", "2000", "Target number of classes, default: 2000")
        ('v', "vocabulary=FILE", "arg", "", "Vocabulary, one word per line")
        ('t', "ll-threshold=FLOAT", "arg", "0.0", "Log likelihood threshold for a split, default: 0.0")
        ('e', "num-split-evals=INT", "arg", "0", "Number of evaluations per split, default: 0")
        ('i', "model-write-interval=INT", "arg", "0", "Interval for writing temporary models, default: 0")
        ('h', "help", "", "", "display help");
        config.default_parse(argc, argv);
        if (config.arguments.size() != 3) config.print_help(stderr, 1);

        string corpus_fname = config.arguments[0];
        string class_fname = config.arguments[1];
        string model_fname = config.arguments[2];

        int num_classes = config["num-classes"].get_int();
        double ll_threshold = config["ll-threshold"].get_float();
        int num_split_evals = config["num-split-evals"].get_int();
        int model_write_interval = config["model-write-interval"].get_int();
        string vocab_fname = config["vocabulary"].get_str();

        Exchange e(corpus_fname, vocab_fname, class_fname);

        time_t t1,t2;
        t1=time(0);
        cerr << "log likelihood: " << e.log_likelihood() << endl;

        vector<set<int> > super_classes;
        map<int, int> super_class_lookup;
        for (int i=0; i<(int)e.m_classes.size(); i++) {
            if (e.m_classes[i].size() > 0) {
                super_class_lookup[i] = super_classes.size();
                set<int> curr_class = { i };
                super_classes.push_back(curr_class);
            }
        }

        split_classes_2(e,
                        num_classes, num_split_evals,
                        ll_threshold,
                        model_fname, model_write_interval,
                        super_classes,
                        super_class_lookup);

        t2=time(0);
        cerr << "Train run time: " << t2-t1 << " seconds" << endl;

        write_super_classes(model_fname + ".superclasses.gz",
                            super_classes,
                            super_class_lookup);

        e.write_word_classes(model_fname + ".cgenprobs.gz");
        e.write_class_mem_probs(model_fname + ".cmemprobs.gz");
        e.write_classes(model_fname + ".classes.gz");

    } catch (string &e) {
        cerr << e << endl;
    }
}
