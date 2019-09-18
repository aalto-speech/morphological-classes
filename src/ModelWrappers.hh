#ifndef MODEL_WRAPPERS
#define MODEL_WRAPPERS

#include <string>

#include "Categories.hh"
#include "CatPerplexity.hh"
#include "Ngram.hh"
#include "defs.hh"

class LanguageModel {
public:
    virtual bool word_in_vocabulary(std::string word) = 0;
    virtual void start_sentence() = 0;
    virtual double likelihood(std::string word) = 0;
    virtual double sentence_end_likelihood() = 0;
    void evaluate(std::string corpus_fname,
                  std::string *probs_fname=nullptr,
                  int *ppl_num_words=nullptr);
};

class WordNgram : public LanguageModel {
public:
    WordNgram(std::string model_filename, bool unk_root_node=false);
    bool word_in_vocabulary(std::string word) override;
    void start_sentence() override;
    double likelihood(std::string word) override;
    double sentence_end_likelihood() override;
private:
    int m_current_node_id;
    bool m_unk_root_node;
    LNNgram m_ln_arpa_model;
};

class ClassNgram : public LanguageModel {
public:
    ClassNgram(
            std::string arpa_filename,
            std::string cmemprobs_filename,
            bool unk_root_node=false);
    bool word_in_vocabulary(std::string word) override;
    void start_sentence() override;
    double likelihood(std::string word) override;
    double sentence_end_likelihood() override;
private:
    int m_current_node_id;
    bool m_unk_root_node;
    LNNgram m_ln_arpa_model;
    std::map<std::string, std::pair<int, flt_type>> m_class_memberships;
    std::vector<int> m_indexmap;
    int m_num_classes;
};

class CategoryNgram : public LanguageModel {
public:
    CategoryNgram(
            std::string arpa_filename,
            std::string cgenprobs_filename,
            std::string cmemprobs_filename,
            bool unk_root_node=false,
            int max_tokens=100,
            double beam=20.0);
    ~CategoryNgram();
    bool word_in_vocabulary(std::string word) override;
    void start_sentence() override;
    double likelihood(std::string word) override;
    double sentence_end_likelihood() override;
private:
    bool m_unk_root_node;
    LNNgram m_ln_arpa_model;
    std::vector<int> m_indexmap;
    Categories m_word_categories;
    CatPerplexity::CategoryHistory *m_history;
    int m_max_tokens;
    int m_beam;
};

class SubwordNgram : public LanguageModel {
public:
    SubwordNgram(std::string model_filename, std::string word_segs_filename);
    bool word_in_vocabulary(std::string word) override;
    void start_sentence() override;
    double likelihood(std::string word) override;
    double sentence_end_likelihood() override;
private:
    void read_word_segs(std::string word_segs_fname, bool only_sws=false);
    int m_current_node_id;
    int m_root_node;
    int m_sentence_start_node;
    LNNgram m_ln_arpa_model;
    std::map<std::string, std::vector<int> > m_word_segs;
};

class InterpolatedLM : public LanguageModel {
public:
    InterpolatedLM(
            LanguageModel *lm1,
            LanguageModel *lm2,
            double interpolation_weight);
    bool word_in_vocabulary(std::string word) override;
    void start_sentence() override;
    double likelihood(std::string word) override;
    double sentence_end_likelihood() override;
private:
    LanguageModel *m_lm1;
    LanguageModel *m_lm2;
    double m_interpolation_weight;
    double m_first_log_iw;
    double m_second_log_iw;
};

#endif /* MODEL_WRAPPERS */
