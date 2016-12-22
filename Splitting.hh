#ifndef SPLITTING
#define SPLITTING

#include <map>
#include <set>
#include <string>
#include <vector>

#include "Merging.hh"


class Splitting : public Merging {
public:
    Splitting();
    Splitting(int num_classes,
              const std::map<std::string, int> &word_classes,
              std::string corpus_fname="");
    ~Splitting() { };

    double evaluate_exchange(int word,
                             int curr_class,
                             int tentative_class) const;
    void do_exchange(int word,
                     int prev_class,
                     int new_class);
    void random_split(const std::set<int> &words,
                      std::set<int> &class1_words,
                      std::set<int> &class2_words) const;
    void do_split(int class_idx, bool random=false);
    int do_split(int class_idx,
                 const std::set<int> &class1_words,
                 const std::set<int> &class2_words);
    double iterate_exchange_local(int class1_idx,
                                  int class2_idx,
                                  int max_exchanges=100000,
                                  int num_threads=1);
    int iterate_exchange_local_2(int class1_idx,
                                 int class2_idx,
                                 int num_iterations=5);
    void local_exchange_thr(int num_threads,
                            int curr_class,
                            int tentative_class,
                            int &best_word,
                            double &best_ll_diff);
    void local_exchange_thr_worker(int num_threads,
                                   int thread_index,
                                   int curr_class,
                                   int tentative_class,
                                   int &best_word,
                                   double &best_ll_diff);
};

#endif /* SPLITTING */
