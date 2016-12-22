#ifndef SPLITTING
#define SPLITTING

#include <map>
#include <set>
#include <string>
#include <vector>


#define START_CLASS 0
#define UNK_CLASS 1


class Splitting {
public:
    Splitting(std::string fname,
              std::string vocab_fname="",
              std::string class_fname="");
    Splitting(int num_classes,
              const std::map<std::string, int> &word_classes,
              std::string fname="",
              std::string vocab_fname="");
    ~Splitting() { };

    void read_corpus(std::string fname);
    void write_class_mem_probs(std::string fname) const;
    void initialize_classes_preset(const std::map<std::string, int> &word_classes);
    int insert_word_to_vocab(std::string word);
    std::map<int,int> read_class_initialization(std::string class_fname);
    void set_class_counts();
    double log_likelihood() const;
    int num_classes() const { return m_num_classes; }
    double evaluate_exchange(int word,
                             int curr_class,
                             int tentative_class) const;
    void do_exchange(int word,
                     int prev_class,
                     int new_class);
    double evaluate_merge(int class1,
                          int class2) const;
    void do_merge(int class1,
                  int class2);
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

    int m_num_classes;
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


#endif /* SPLITTING */

