#ifndef CATEGORIES
#define CATEGORIES

#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>

#include "io.hh"
#include "defs.hh"
#include "Ngram.hh"

class TrainingParameters {
public:
    TrainingParameters()
        : num_tokens(100),
          num_final_tokens(10),
          num_parses(0),
          max_order(3),
          max_line_length(100),
          prob_beam(10.0),
          verbose(false) { };

    unsigned int num_tokens;
    unsigned int num_final_tokens;
    unsigned int num_parses;
    unsigned int max_order;
    unsigned int max_line_length;
    flt_type prob_beam;
    bool verbose;
};


class Token {
public:
    Token() : m_category(-1),
              m_cng_node(-1),
              m_lp(0.0),
              m_gen_lp(0.0),
              m_prev_token(nullptr) { };

    Token(Token *prev_token,
          int category,
          flt_type score)
    {
        m_category = category;
        m_cng_node = prev_token->m_cng_node;
        m_lp = score;
        m_gen_lp = 0.0;
        m_prev_token = prev_token;
    }

    Token(Token &prev_token,
          int category)
    {
        m_category = category;
        m_cng_node = prev_token.m_cng_node;
        m_lp = prev_token.m_lp;
        m_gen_lp = 0.0;
        m_prev_token = &prev_token;
    }

    ~Token() { };

    int m_category;
    int m_cng_node;
    flt_type m_lp;
    flt_type m_gen_lp;
    Token *m_prev_token;
};


typedef std::map<int, flt_type> CategoryProbs;

class Categories {
public:
    Categories() { m_num_categories = 0; };
    Categories(int num_categories);
    Categories(std::string filename,
               const std::map<std::string, int> &counts,
               int top_word_categories=0);
    void accumulate(std::string word, int c, flt_type weight);
    void accumulate(Categories &acc);
    void estimate_model();
    int num_words() const;
    int num_words_with_categories() const;
    int num_categories() const;
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

    int m_num_categories;

    // Sufficient statistics
    std::map<std::string, CategoryProbs> m_stats;

    // Final model p(c|w)
    std::map<std::string, CategoryProbs> m_category_gen_probs;
    // Final model p(w|c)
    std::map<std::string, CategoryProbs> m_category_mem_probs;
};


void segment_sent(const std::vector<std::string> &sent,
                  const Ngram &ngram,
                  const std::vector<int> &indexmap,
                  const Categories &categories,
                  TrainingParameters &params,
                  std::vector<std::vector<Token*> > &tokens,
                  std::vector<Token*> &pointers,
                  unsigned long int *num_vocab_words=nullptr,
                  unsigned long int *num_oov_words=nullptr,
                  unsigned long int *num_unpruned_tokens=nullptr,
                  unsigned long int *num_pruned_tokens=nullptr);

flt_type collect_stats(const std::vector<std::string> &sent,
                       const Ngram &ngram,
                       const std::vector<int> &indexmap,
                       const Categories &categories,
                       Categories &stats,
                       SimpleFileOutput *seqf,
                       TrainingParameters &params,
                       unsigned long int *num_vocab_words=nullptr,
                       unsigned long int *num_oov_words=nullptr,
                       unsigned long int *num_unpruned_tokens=nullptr,
                       unsigned long int *num_pruned_tokens=nullptr);

void limit_num_categories(std::map<std::string, CategoryProbs> &probs,
                          int num_categories);

void histogram_prune(std::vector<Token*> &tokens,
                     int num_tokens,
                     flt_type worst_score,
                     flt_type best_score);


#endif /* CATEGORIES */
