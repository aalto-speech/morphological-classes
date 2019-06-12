#include "ModelWrappers.hh"
#include "defs.hh"

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
