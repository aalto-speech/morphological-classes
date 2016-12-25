#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <ctime>
#include <cfloat>

#include "io.hh"
#include "defs.hh"
#include "conf.hh"
#include "Splitting.hh"

using namespace std;


struct SplitEvalTask {
    SplitEvalTask() : cidx(-1), ll(-DBL_MAX) { }
    int cidx;
    double ll;
    set<int> class1_words;
    set<int> class2_words;
    vector<int> ordered_words;
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


void find_candidate_classes(Splitting &spl,
                            vector<int> &classes_to_evaluate,
                            int num_classes)
{
    double max_word_types=0.0;
    double max_word_tokens=0.0;
    for (int i=spl.m_num_special_classes; i<(int)spl.m_classes.size(); i++) {
        max_word_types = max(max_word_types, (double)spl.m_classes[i].size());
        max_word_tokens = max(max_word_tokens, (double)spl.m_class_counts[i]);
    }

    multimap<double, int> class_order;
    for (int i=spl.m_num_special_classes; i<(int)spl.m_classes.size(); i++) {
        if (spl.m_classes[i].size() < 2) continue;
        double score = 0.5 * (double)spl.m_classes[i].size() / max_word_types;
        score += 0.5 * (double)spl.m_class_counts[i] / max_word_tokens;
        class_order.insert(make_pair(score, i));
    }

    auto coit = class_order.rbegin();
    classes_to_evaluate.clear();
    while ((int)classes_to_evaluate.size() < num_classes) {
        classes_to_evaluate.push_back(coit->second);
        coit++;
    }
}


void split_classes(Splitting &spl,
                   int target_num_classes,
                   int num_eval_classes,
                   double ll_threshold,
                   string model_fname,
                   int model_write_interval,
                   vector<set<int> > &super_classes,
                   map<int, int> &super_class_lookup)
{
    while (spl.num_classes() < target_num_classes)
    {
        vector<int> classes_to_evaluate;
        find_candidate_classes(spl, classes_to_evaluate, num_eval_classes);
        SplitEvalTask best_split;
        best_split.cidx = classes_to_evaluate[0];

        if (num_eval_classes < 2) {
            cerr << "split class " << best_split.cidx << ", size: " << spl.m_classes[best_split.cidx].size() << endl;
            spl.freq_split(spl.m_classes[best_split.cidx],
                         best_split.class1_words, best_split.class2_words,
                         best_split.ordered_words);
        }
        else {
            for (int ec=0; ec<num_eval_classes; ec++) {
                SplitEvalTask split_task;
                split_task.cidx = classes_to_evaluate[ec];
                spl.freq_split(spl.m_classes[split_task.cidx],
                             split_task.class1_words, split_task.class2_words, split_task.ordered_words);
                double orig_ll = spl.log_likelihood();
                int class2_idx = spl.do_split(split_task.cidx, split_task.class1_words, split_task.class2_words);
                spl.iterate_exchange_local(split_task.cidx, class2_idx, split_task.ordered_words, 1);
                double split_ll = spl.log_likelihood();
                split_task.ll = split_ll - orig_ll;
                if (split_task.ll > best_split.ll)
                    best_split = split_task;
                spl.do_merge(split_task.cidx, class2_idx);
            }
        }

        cerr << "splitting.." << endl;
        int class2_idx = spl.do_split(best_split.cidx, best_split.class1_words, best_split.class2_words);
        cerr << spl.num_classes() << "\t" << spl.log_likelihood() << endl;
        cerr << "running local exchange algorithm.." << endl;
        spl.iterate_exchange_local(best_split.cidx, class2_idx, best_split.ordered_words);
        cerr << "final class sizes: " << spl.m_classes[best_split.cidx].size() << " "
                << spl.m_classes[class2_idx].size() << endl;
        cerr << spl.num_classes() << "\t" << spl.log_likelihood() << endl;

        int sci = super_class_lookup[best_split.cidx];
        super_classes[sci].insert(class2_idx);
        super_class_lookup[class2_idx] = sci;

        if (model_write_interval > 0 && spl.num_classes() % model_write_interval == 0) {
            write_super_classes(model_fname + "." + int2str(spl.num_classes()) + ".superclasses.gz",
                                super_classes,
                                super_class_lookup);
            spl.write_class_mem_probs(model_fname + "." + int2str(spl.num_classes()) + ".cmemprobs.gz");
        }
    }
}


int main(int argc, char* argv[])
{
    try {
        conf::Config config;
        config("usage: split [OPTION...] CORPUS CLASS_INIT MODEL\n")
        ('c', "num-classes=INT", "arg", "2000", "Target number of classes, default: 2000")
        ('t', "ll-threshold=FLOAT", "arg", "0.0", "Log likelihood threshold for a split, default: 0.0")
        ('e', "num-split-evals=INT", "arg", "0", "Number of evaluations per split, default: 0")
        ('i', "model-write-interval=INT", "arg", "0", "Interval for writing temporary models, default: 0")
        ('h', "help", "", "", "display help");
        config.default_parse(argc, argv);
        if (config.arguments.size() != 3) config.print_help(stderr, 1);

        string corpus_fname = config.arguments[0];
        string class_init_fname = config.arguments[1];
        string model_fname = config.arguments[2];

        int num_classes = config["num-classes"].get_int();
        double ll_threshold = config["ll-threshold"].get_float();
        int num_split_evals = config["num-split-evals"].get_int();
        int model_write_interval = config["model-write-interval"].get_int();

        Splitting spl;
        map<int,int> class_idx_mapping = spl.read_class_initialization(class_init_fname);
        spl.read_corpus(corpus_fname);

        time_t t1,t2;
        t1=time(0);
        cerr << "log likelihood: " << spl.log_likelihood() << endl;

        vector<set<int> > super_classes;
        map<int, int> super_class_lookup;
        for (int i=0; i<(int)spl.m_classes.size(); i++) {
            if (spl.m_classes[i].size() > 0) {
                super_class_lookup[i] = super_classes.size();
                set<int> curr_class = { i };
                super_classes.push_back(curr_class);
            }
        }

        split_classes(spl,
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

        spl.write_class_mem_probs(model_fname + ".cmemprobs.gz");

    } catch (string &e) {
        cerr << e << endl;
    }
}
