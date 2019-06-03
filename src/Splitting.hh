#ifndef SPLITTING
#define SPLITTING

#include <map>
#include <set>
#include <string>
#include <vector>

#include "Exchanging.hh"

class Splitting : public Exchanging {
public:
    Splitting();
    Splitting(int num_classes,
            const std::map<std::string, int>& word_classes,
            std::string corpus_fname = "");
    ~Splitting() { };

    void freq_split(const std::set<int>& words,
            std::set<int>& class1_words,
            std::set<int>& class2_words,
            std::vector<int>& ordered_words) const;
    int do_split(int class_idx,
            const std::set<int>& class1_words,
            const std::set<int>& class2_words);
    int iterate_exchange_local(int class1_idx,
            int class2_idx,
            std::vector<int>& ordered_words,
            int num_iterations = 5);
};

#endif /* SPLITTING */
