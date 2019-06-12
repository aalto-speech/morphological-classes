#include "ModelWrappers.hh"

using namespace std;

WordNgram::WordNgram(
        std::string arpa_filename,
        bool unk_root_node)
{
    m_ln_arpa_model.read_arpa(arpa_filename);
    m_unk_root_node = unk_root_node;
    start_sentence();
}

bool
WordNgram::word_in_vocabulary(std::string word)
{
    if (word==UNK_SYMBOL) return false;
    if (word==CAP_UNK_SYMBOL) return false;
    return m_ln_arpa_model.vocabulary_lookup.find(word)!=m_ln_arpa_model.vocabulary_lookup.end();
}

void
WordNgram::start_sentence()
{
    m_current_node_id = m_ln_arpa_model.sentence_start_node;
}

double
WordNgram::likelihood(std::string word)
{
    double ln_log_prob = 0.0;
    if (word_in_vocabulary(word)) {
        int sym = m_ln_arpa_model.vocabulary_lookup[word];
        m_current_node_id = m_ln_arpa_model.score(m_current_node_id, sym, ln_log_prob);
    }
    else {
        // SRILM
        if (m_unk_root_node)
            m_current_node_id = m_ln_arpa_model.root_node;
        // VariKN style UNKs
        else
            m_current_node_id = m_ln_arpa_model.advance(m_current_node_id, m_ln_arpa_model.unk_symbol_idx);
    }
    return ln_log_prob;
}

double
WordNgram::sentence_end_likelihood()
{
    return likelihood(m_ln_arpa_model.sentence_end_symbol);
}


ClassNgram::ClassNgram(
        std::string arpa_filename,
        std::string cmemprobs_filename,
        bool unk_root_node)
{
    m_ln_arpa_model.read_arpa(arpa_filename);
    m_unk_root_node = unk_root_node;
    m_num_classes = read_class_memberships(cmemprobs_filename, m_class_memberships);
    m_indexmap = get_class_index_map(m_num_classes, m_ln_arpa_model);
    start_sentence();
}

bool
ClassNgram::word_in_vocabulary(std::string word)
{
    if (word==UNK_SYMBOL) return false;
    if (word==CAP_UNK_SYMBOL) return false;
    return m_class_memberships.find(word)!=m_class_memberships.end();
}

void
ClassNgram::start_sentence()
{
    m_current_node_id = m_ln_arpa_model.sentence_start_node;
}

double
ClassNgram::likelihood(std::string word)
{
    double ln_log_prob = 0.0;
    if (word_in_vocabulary(word)) {
        pair<int, flt_type> word_class = m_class_memberships.at(word);
        ln_log_prob = word_class.second;
        m_current_node_id = m_ln_arpa_model.score(m_current_node_id, m_indexmap[word_class.first], ln_log_prob);
    }
    else {
        // SRILM
        if (m_unk_root_node)
            m_current_node_id = m_ln_arpa_model.root_node;
            // VariKN style UNKs
        else
            m_current_node_id = m_ln_arpa_model.advance(m_current_node_id, m_ln_arpa_model.unk_symbol_idx);
    }
    return ln_log_prob;
}

double
ClassNgram::sentence_end_likelihood()
{
    double ln_log_prob = 0.0;
    m_current_node_id = m_ln_arpa_model.score(m_current_node_id, m_ln_arpa_model.sentence_end_symbol_idx, ln_log_prob);
    return ln_log_prob;
}
