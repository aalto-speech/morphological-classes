#ifndef EXCHANGING
#define EXCHANGING

#include <map>
#include <set>
#include <string>
#include <vector>

#include "Merging.hh"


class Exchanging : public Merging {
public:
    Exchanging();
    Exchanging(int num_classes,
               std::string corpus_fname,
               std::string vocab_fname="",
               bool old_init=false);
    Exchanging(int num_classes,
               const std::map<std::string, int> &word_classes,
               std::string corpus_fname="");
    ~Exchanging() { };

    void initialize_classes_by_freq(std::string corpus_fname,
                                    std::string vocab_fname);
    void initialize_classes_by_freq_2(std::string corpus_fname,
                                      std::string vocab_fname);
    double evaluate_exchange(int word,
                             int curr_class,
                             int tentative_class) const;
    void do_exchange(int word,
                     int prev_class,
                     int new_class);
    double iterate_exchange(int max_iter=0,
                            int max_seconds=0,
                            int ll_print_interval=0,
                            int model_write_interval=0,
                            std::string model_base="",
                            int num_threads=1);
    double iterate_exchange(std::vector<std::vector<int> > super_classes,
                            std::map<int, int> super_class_lookup,
                            int max_iter=0,
                            int max_seconds=0,
                            int ll_print_interval=0,
                            int model_write_interval=0,
                            std::string model_base="");
    void exchange_thr(int num_threads,
                      int word_index,
                      int curr_class,
                      int &best_class,
                      double &best_ll_diff);
    void exchange_thr_worker(int num_threads,
                             int thread_index,
                             int word_index,
                             int curr_class,
                             int &best_class,
                             double &best_ll_diff);

};

#endif /* EXCHANGING */
