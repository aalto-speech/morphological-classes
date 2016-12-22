#ifndef MERGING
#define MERGING

#include <map>
#include <set>
#include <string>
#include <vector>


#define START_CLASS 0
#define UNK_CLASS 1


class Merging {
public:
    Merging(std::string fname,
            std::string vocab_fname="",
            std::string class_fname="",
            unsigned int top_word_classes=0);
    Merging(int num_classes,
            const std::map<std::string, int> &word_classes,
            std::string fname="",
            std::string vocab_fname="");
    ~Merging() { };

    void read_corpus(std::string fname,
                     std::string vocab_fname="");
    void write_class_mem_probs(std::string fname) const;
    void write_classes(std::string fname) const;
    void initialize_classes_preset(const std::map<std::string, int> &word_classes);
    void read_class_initialization(std::string class_fname);
    void set_class_counts();
    double log_likelihood() const;
    int num_classes() const { return m_num_classes; }
    void do_exchange(int word,
                     int prev_class,
                     int new_class);
    double evaluate_merge(int class1,
                          int class2) const;
    void do_merge(int class1,
                  int class2);

//private:

    int m_num_classes;
    bool m_word_boundary;
    int m_num_special_classes;

    std::vector<std::string> m_vocabulary;
    std::map<std::string, int> m_vocabulary_lookup;

    std::vector<std::set<int> > m_classes;
    std::vector<int> m_word_classes;

    std::vector<int> m_word_counts;
    std::vector<std::map<int, int> > m_word_bigram_counts;
    std::vector<std::map<int, int> > m_word_rev_bigram_counts;

    std::vector<int> m_class_counts;
    std::vector<std::vector<int> > m_class_bigram_counts;

    std::vector<std::map<int, int> > m_class_word_counts; // First index word, second source class
    std::vector<std::map<int, int> > m_word_class_counts; // First index word, second target class
};


#endif /* MERGING */

