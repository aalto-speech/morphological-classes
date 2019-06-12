#ifndef MODEL_WRAPPERS
#define MODEL_WRAPPERS

#include <string>

#include "Ngram.hh"
#include "defs.hh"

class LanguageModel {
public:
    virtual bool word_in_vocabulary(std::string word) = 0;
    virtual void start_sentence() = 0;
    virtual double likelihood(std::string word) = 0;
    virtual double sentence_end_likelihood() = 0;
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

#endif /* MODEL_WRAPPERS */