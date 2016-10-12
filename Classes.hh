#ifndef CLASSES
#define CLASSES

#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>

#include "io.hh"
#include "defs.hh"


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


typedef std::map<int, flt_type> WordClassProbs;

class WordClasses {
public:
    WordClasses() { m_num_classes = 0; };
    WordClasses(int num_classes);
    WordClasses(std::string filename,
                const std::map<std::string, int> &counts,
                int top_word_classes=0);
    void accumulate(std::string word, int c, flt_type weight);
    void accumulate(WordClasses &acc);
    void estimate_model();
    int num_words() const;
    int num_words_with_classes() const;
    int num_classes() const;
    int num_observed_classes() const;
    int num_class_probs() const;
    int num_word_probs() const;
    int num_stats() const;
    void get_words(std::set<std::string> &words, bool get_unanalyzed=true);
    void get_unanalyzed_words(std::set<std::string> &words);
    void get_unanalyzed_words(std::map<std::string, flt_type> &words);
    flt_type log_likelihood(int c, std::string word) const;
    flt_type log_likelihood(int c, const WordClassProbs *wcp) const;
    const WordClassProbs* get_word_probs(std::string word) const;
    const WordClassProbs* get_class_probs(std::string word) const;
    void get_all_word_probs(std::vector<std::map<std::string, flt_type> > &word_probs) const;
    bool assert_class_probs() const;
    bool assert_word_probs() const;
    void write_class_probs(std::string fname) const;
    void write_word_probs(std::string fname) const;
    void read_class_probs(std::string fname);
    void read_word_probs(std::string fname);

    int m_num_classes;

    // Sufficient statistics
    std::map<std::string, WordClassProbs> m_stats;

    // Final model p(c|w)
    std::map<std::string, WordClassProbs> m_class_gen_probs;
    // Final model p(w|c)
    std::map<std::string, WordClassProbs> m_class_memberships;
};


typedef std::map<int, flt_type> NgramCtxt;

class ClassNgram
{
public:
    virtual ~ClassNgram() { };
    virtual const NgramCtxt* get_context(int c2, int c1) const = 0;
    virtual flt_type log_likelihood(const NgramCtxt *ctxt, int c) const = 0;
    virtual void accumulate(std::vector<int> &classes, flt_type weight) = 0;
    virtual void accumulate(const ClassNgram *acc) = 0;
    virtual void estimate_model(bool discard_unks=false) = 0;
    virtual void write_model(std::string fname) const = 0;
    virtual void read_model(std::string fname) = 0;
    virtual bool assert_model() = 0;
    virtual int num_grams() const = 0;
    virtual ClassNgram* get_new() const = 0;
};


class Unigram : public ClassNgram
{
public:
    Unigram() { };
    Unigram(int num_classes);
    ~Unigram() { };
    const NgramCtxt* get_context(int c2, int c1) const;
    flt_type log_likelihood(const NgramCtxt *ctxt, int c) const;
    void accumulate(std::vector<int> &classes, flt_type weight);
    void accumulate(const ClassNgram *acc);
    void estimate_model(bool discard_unks=false);
    void write_model(std::string fname) const;
    void read_model(std::string fname);
    bool assert_model();
    int num_grams() const;
    ClassNgram* get_new() const;

    NgramCtxt m_unigrams;
};


class Bigram : public ClassNgram
{
public:
    Bigram() { };
    ~Bigram() { };
    const NgramCtxt* get_context(int c2, int c1) const;
    flt_type log_likelihood(const NgramCtxt *ctxt, int c) const;
    void accumulate(std::vector<int> &classes, flt_type weight);
    void accumulate(const ClassNgram *acc);
    void estimate_model(bool discard_unks=false);
    void write_model(std::string fname) const;
    void read_model(std::string fname);
    bool assert_model();
    int num_grams() const;
    ClassNgram* get_new() const;

    std::map<int, NgramCtxt> m_bigrams;
};


class Trigram : public ClassNgram
{
public:
    Trigram() { };
    ~Trigram() { };
    const NgramCtxt* get_context(int c2, int c1) const;
    flt_type log_likelihood(const NgramCtxt *ctxt, int c) const;
    void accumulate(std::vector<int> &classes, flt_type weight);
    void accumulate(const ClassNgram *acc);
    void estimate_model(bool discard_unks=false);
    void write_model(std::string fname) const;
    void read_model(std::string fname);
    bool assert_model();
    int num_grams() const;
    ClassNgram* get_new() const;

    std::map<int, std::map<int, NgramCtxt> > m_trigrams;
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
                  const ClassNgram *ngram,
                  const WordClasses *word_classes,
                  flt_type prob_beam,
                  unsigned int max_tokens,
                  unsigned int max_final_tokens,
                  unsigned long int &unpruned,
                  unsigned long int &pruned,
                  std::vector<std::vector<Token*> > &tokens,
                  std::vector<Token*> &pointers);

flt_type collect_stats(const std::vector<std::vector<std::string> > &sents,
                       const ClassNgram *ngram,
                       const WordClasses *word_classes,
                       ClassNgram *ngram_stats,
                       WordClasses *word_stats,
                       unsigned int max_tokens=100,
                       unsigned int max_final_tokens=10,
                       unsigned int num_threads=0,
                       unsigned int thread_index=0,
                       flt_type *retval=nullptr,
                       int *skipped_sents=nullptr,
                       flt_type prob_beam=10.0,
                       bool verbose=false);

flt_type collect_stats_thr(const std::vector<std::vector<std::string> > &sents,
                           const ClassNgram *ngram,
                           const WordClasses *word_classes,
                           ClassNgram *ngram_stats,
                           WordClasses *word_stats,
                           unsigned int num_threads,
                           unsigned int max_tokens=100,
                           unsigned int max_final_tokens=10,
                           flt_type prob_beam=10.0,
                           bool verbose=false);

void print_class_seqs(std::string &fname,
                      const std::vector<std::vector<std::string> > &sents,
                      const ClassNgram *ngram,
                      const WordClasses *word_classes,
                      unsigned int max_tokens=100,
                      flt_type prob_beam=100.0,
                      unsigned int max_parses=10);

void print_class_seqs(SimpleFileOutput &seqf,
                      const std::vector<std::vector<std::string> > &sents,
                      const ClassNgram *ngram,
                      const WordClasses *word_classes,
                      unsigned int max_tokens=100,
                      flt_type prob_beam=100.0,
                      unsigned int max_parses=10);

void limit_num_classes(std::map<std::string, WordClassProbs> &probs,
                       int num_classes);

void histogram_prune(std::vector<Token*> &tokens,
                     int num_tokens,
                     flt_type worst_score,
                     flt_type best_score);


#endif /* CLASSES */
