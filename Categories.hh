#ifndef CLASSES
#define CLASSES

#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>

#include "io.hh"
#include "defs.hh"
#include "Ngram.hh"


class Token {
public:
    Token() {
        m_class = START_CLASS;
        m_score = 0.0;
        m_prev_token = nullptr;
    }
    Token(Token *prev_token, int clss, flt_type score) {
        m_prev_token = prev_token;
        m_class = clss;
        m_score = score;
    }
    Token(Token &prev_token, int clss) {
        m_prev_token = &prev_token;
        m_class = clss;
        m_score = prev_token.m_score;
    }
    ~Token() { };

    int m_class;
    flt_type m_score;
    Token *m_prev_token;
};


typedef std::map<int, flt_type> CategoryProbs;

class Categories {
public:
    Categories() { m_num_classes = 0; };
    Categories(int num_classes);
    Categories(std::string filename,
               const std::map<std::string, int> &counts,
               int top_word_categories=0);
    void accumulate(std::string word, int c, flt_type weight);
    void accumulate(Categories &acc);
    void estimate_model();
    int num_words() const;
    int num_words_with_categories() const;
    int num_classes() const;
    int num_observed_categories() const;
    int num_category_gen_probs() const;
    int num_category_mem_probs() const;
    int num_stats() const;
    void get_words(std::set<std::string> &words, bool get_unanalyzed=true);
    void get_unanalyzed_words(std::set<std::string> &words);
    void get_unanalyzed_words(std::map<std::string, flt_type> &words);
    flt_type log_likelihood(int c, std::string word) const;
    flt_type log_likelihood(int c, const CategoryProbs *wcp) const;
    const CategoryProbs* get_category_gen_probs(std::string word) const;
    const CategoryProbs* get_category_mem_probs(std::string word) const;
    void get_all_category_mem_probs(std::vector<std::map<std::string, flt_type> > &word_probs) const;
    bool assert_category_gen_probs() const;
    bool assert_category_mem_probs() const;
    void write_category_gen_probs(std::string fname) const;
    void write_category_mem_probs(std::string fname) const;
    void read_category_gen_probs(std::string fname);
    void read_category_mem_probs(std::string fname);

    int m_num_classes;

    // Sufficient statistics
    std::map<std::string, CategoryProbs> m_stats;

    // Final model p(c|w)
    std::map<std::string, CategoryProbs> m_class_gen_probs;
    // Final model p(w|c)
    std::map<std::string, CategoryProbs> m_class_mem_probs;
};


int get_word_counts(std::string corpusfname,
                    std::map<std::string, int> &counts);

int read_sents(std::string corpusfname,
               std::vector<std::vector<std::string> > &sents,
               int maxlen=20,
               std::set<std::string> *vocab=nullptr,
               int *num_word_tokens=nullptr,
               int *num_word_types=nullptr,
               int *num_unk_tokens=nullptr,
               int *num_unk_types=nullptr);

void segment_sent(const std::vector<std::string> &sent,
                  const Ngram &ngram,
                  const Categories &categories,
                  flt_type prob_beam,
                  unsigned int max_tokens,
                  unsigned int max_final_tokens,
                  unsigned long int &unpruned,
                  unsigned long int &pruned,
                  std::vector<std::vector<Token*> > &tokens,
                  std::vector<Token*> &pointers);

flt_type collect_stats(std::vector<std::string> &sent,
                       const Ngram &ngram,
                       const Categories &categories,
                       Categories &stats,
                       SimpleFileOutput &seqf,
                       unsigned int num_tokens=100,
                       unsigned int num_final_tokens=10,
                       unsigned int num_parses=0,
                       flt_type prob_beam=10.0,
                       bool verbose=false);

void limit_num_classes(std::map<std::string, CategoryProbs> &probs,
                       int num_classes);

void histogram_prune(std::vector<Token*> &tokens,
                     int num_tokens,
                     flt_type worst_score,
                     flt_type best_score);


#endif /* CLASSES */
