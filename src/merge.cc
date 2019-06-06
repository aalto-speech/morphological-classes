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
#include "Merging.hh"

using namespace std;

void read_super_classes(
        string scfname,
        map<int, int>& class_idx_mapping,
        vector<vector<int>>& super_classes,
        map<int, int>& super_class_lookup)
{
    cerr << "Reading super class definitions from " << scfname << endl;
    super_classes.clear();
    super_class_lookup.clear();

    SimpleFileInput classf(scfname);
    string line;
    while (classf.getline(line)) {
        if (!line.length()) continue;
        stringstream liness(line);
        string token;
        vector<int> curr_super_class;
        while (std::getline(liness, token, ',')) {
            int class_idx = str2int(token);
            class_idx = class_idx_mapping[class_idx];
            curr_super_class.push_back(class_idx);
            super_class_lookup[class_idx] = super_classes.size();
        }
        super_classes.push_back(curr_super_class);
    }
}

struct MergeEvalTask {
  MergeEvalTask()
          :super_class_idx(-1), c1idx(-1), c2idx(-1), idx_to_remove(-1), ll(-DBL_MAX) { }
  int super_class_idx;
  int c1idx;
  int c2idx;
  int idx_to_remove;
  double ll;
};

void
merge_thr_worker(const Merging& merging,
        int num_threads,
        int thread_index,
        const vector<MergeEvalTask>& evals,
        MergeEvalTask& best_merge)
{
    for (int i = 0; i<(int) evals.size(); i++) {
        if (i%num_threads!=thread_index) continue;
        const MergeEvalTask& task = evals[i];
        double merge_ll = merging.evaluate_merge(task.c1idx, task.c2idx);
        if (merge_ll>best_merge.ll) {
            best_merge.ll = merge_ll;
            best_merge.c1idx = task.c1idx;
            best_merge.c2idx = task.c2idx;
            best_merge.idx_to_remove = task.idx_to_remove;
            best_merge.super_class_idx = task.super_class_idx;
        }
    }
}

void merge_classes(Merging& merging,
        vector<vector<int>>& super_classes,
        map<int, int>& super_class_lookup,
        int target_num_classes,
        int evals_per_iteration,
        int num_threads,
        string model_fname,
        int model_write_interval)
{
    srand(0);

    while (merging.num_classes()>target_num_classes) {
        vector<MergeEvalTask> eval_tasks;

        for (int sci = 0; sci<(int) super_classes.size(); ++sci) {
            vector<int>& super_class = super_classes[sci];
            if (super_class.size()<2) continue;

            int evals_per_super_class = std::max(1, (int) round(double(super_class.size())/double(merging.num_classes())
                    *evals_per_iteration));

            for (int i = 0; i<evals_per_super_class; i++) {
                int idx1 = rand()%super_class.size();
                int idx2 = rand()%super_class.size();

                if (idx1==idx2) continue;
                if (idx1>idx2) swap(idx1, idx2);

                int c1idx = super_class[idx1];
                int c2idx = super_class[idx2];

                if (merging.m_classes[c1idx].size()==0) continue;
                if (merging.m_classes[c2idx].size()==0) continue;

                MergeEvalTask task;
                task.super_class_idx = sci;
                task.c1idx = c1idx;
                task.c2idx = c2idx;
                task.idx_to_remove = idx2;
                task.ll = 0.0;
                eval_tasks.push_back(task);
            }
        }

        vector<MergeEvalTask> thr_best_tasks(num_threads);
        MergeEvalTask best_task;
        vector<std::thread*>workers;
        for (int t = 0; t<num_threads; t++) {
            std::thread* worker = new std::thread(&merge_thr_worker,
                    std::ref(merging),
                    num_threads, t,
                    std::ref(eval_tasks),
                    std::ref(thr_best_tasks[t]));
            workers.push_back(worker);
        }
        for (int t = 0; t<num_threads; t++) {
            workers[t]->join();
            if (thr_best_tasks[t].ll>best_task.ll)
                best_task = thr_best_tasks[t];
        }

        merging.do_merge(best_task.c1idx, best_task.c2idx);
        int msci = super_class_lookup[best_task.c2idx];
        assert(*(super_classes[msci].begin()+best_task.idx_to_remove)==best_task.idx_to_remove);
        super_classes[msci].erase(super_classes[msci].begin()+best_task.idx_to_remove);
        cerr << merging.num_classes() << "\t" << merging.log_likelihood() << endl;

        if (model_write_interval>0 && merging.num_classes()%model_write_interval==0) {
            merging.write_class_mem_probs(model_fname+"."+int2str(merging.num_classes())+".cmemprobs.gz");
        }
    }
}

int main(int argc, char* argv[])
{
    try {
        conf::Config config;
        config("usage: merge [OPTION...] CORPUS CLASS_INIT SUPER_CLASSES MODEL\n")
                ('c', "num-classes=INT", "arg", "1000", "Target number of classes, default: 1000")
                ('t', "num-threads=INT", "arg", "1", "Number of threads, default: 1")
                ('m', "num-merge-evals=INT", "arg", "1000", "Number of evaluations per merge, default: 1000")
                ('i', "model-write-interval=INT", "arg", "0", "Interval for writing temporary models, default: 0")
                ('h', "help", "", "", "display help");
        config.default_parse(argc, argv);
        if (config.arguments.size()!=4) config.print_help(stderr, 1);

        string corpus_fname = config.arguments[0];
        string class_init_fname = config.arguments[1];
        string super_class_fname = config.arguments[2];
        string model_fname = config.arguments[3];

        int num_classes = config["num-classes"].get_int();
        int num_threads = config["num-threads"].get_int();
        int num_merge_evals = config["num-merge-evals"].get_int();
        int model_write_interval = config["model-write-interval"].get_int();

        Merging mrg;
        map<int, int> class_idx_mapping = mrg.read_class_initialization(class_init_fname);
        mrg.read_corpus(corpus_fname);

        vector<vector<int>> super_classes;
        map<int, int> super_class_lookup;
        read_super_classes(super_class_fname, class_idx_mapping,
                super_classes, super_class_lookup);

        time_t t1, t2;
        t1 = time(0);
        cerr << "log likelihood: " << mrg.log_likelihood() << endl;

        merge_classes(mrg, super_classes, super_class_lookup,
                num_classes, num_merge_evals, num_threads,
                model_fname, model_write_interval);

        t2 = time(0);
        cerr << "Train run time: " << t2-t1 << " seconds" << endl;

        mrg.write_class_mem_probs(model_fname+".cmemprobs.gz");

    }
    catch (string& e) {
        cerr << e << endl;
    }
}
