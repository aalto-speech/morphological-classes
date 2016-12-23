#include <string>
#include <sstream>
#include <iostream>
#include <ctime>

#include "io.hh"
#include "defs.hh"
#include "conf.hh"
#include "Exchanging.hh"

using namespace std;


void read_super_classes(string scfname,
                        vector<vector<int> > &super_classes,
                        map<int, int> &super_class_lookup)
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
        while(std::getline(liness, token, ',')) {
            int class_idx = str2int(token);
            curr_super_class.push_back(class_idx);
            super_class_lookup[class_idx] = super_classes.size();
        }
        super_classes.push_back(curr_super_class);
    }
}


int main(int argc, char* argv[])
{
    try {
        conf::Config config;
        config("usage: exchange [OPTION...] CORPUS CLASS_INIT MODEL\n")
        ('a', "max-iter=INT", "arg", "100", "Maximum number of iterations, default: 100")
        ('m', "max-time=INT", "arg", "100000", "Optimization time limit, default: 100000 (seconds)")
        ('t', "num-threads=INT", "arg", "1", "Number of threads, default: 1")
        ('p', "ll-print-interval=INT", "arg", "100000", "Likelihood print interval, default: 100000 (words)")
        ('w', "model-write-interval=INT", "arg", "3600", "Model write interval, default: 3600 (seconds)")
        ('s', "super-classes=FILE", "arg", "", "Superclass definitions")
        ('h', "help", "", "", "display help");
        config.default_parse(argc, argv);
        if (config.arguments.size() != 3) config.print_help(stderr, 1);

        string corpus_fname = config.arguments[0];
        string class_init_fname = config.arguments[1];
        string model_fname = config.arguments[2];

        int max_iter = config["max-iter"].get_int();
        int max_seconds = config["max-time"].get_int();
        int ll_print_interval = config["ll-print-interval"].get_int();
        int num_threads = config["num-threads"].get_int();
        int model_write_interval = config["model-write-interval"].get_int();

        if (config["super-classes"].specified && !config["class-init"].specified) {
            cerr << "Superclass definitions are only usable with a class initialization" << endl;
            exit(1);
        }

        Exchanging exc;
        map<int,int> class_idx_mapping = exc.read_class_initialization(class_init_fname);
        exc.read_corpus(corpus_fname);

        time_t t1,t2;
        t1=time(0);
        cerr << "log likelihood: " << exc.log_likelihood() << endl;

        if (config["super-classes"].specified) {
            vector<vector<int> > super_classes;
            map<int, int> super_class_lookup;
            read_super_classes(config["super-classes"].get_str(), super_classes, super_class_lookup);
            exc.iterate_exchange(super_classes, super_class_lookup,
                                 max_iter, max_seconds, ll_print_interval,
                                 model_write_interval,
                                 model_fname);
        }
        else {
            exc.iterate_exchange(max_iter, max_seconds, ll_print_interval,
                                  model_write_interval, model_fname, num_threads);
        }

        t2=time(0);
        cerr << "Train run time: " << t2-t1 << " seconds" << endl;

        exc.write_class_mem_probs(model_fname + ".cmemprobs.gz");

    } catch (string &e) {
        cerr << e << endl;
    }
}
