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
    void freq_split(const std::set<int> &words,
                    std::set<int> &class1_words,
                    std::set<int> &class2_words,
                    std::vector<int> &ordered_words) const;
    int do_split(int class_idx,
                 const std::set<int> &class1_words,
                 const std::set<int> &class2_words);
    int iterate_exchange_local(int class1_idx,
                               int class2_idx,
                               std::vector<int> &ordered_words,
                               int num_iterations=5);
};

#endif /* SPLITTING */
