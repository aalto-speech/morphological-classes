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


CategoryNgram::CategoryNgram(
        std::string arpa_filename,
        std::string cgenprobs_filename,
        std::string cmemprobs_filename,
        bool unk_root_node,
        int max_tokens,
        double beam)
{
    m_ln_arpa_model.read_arpa(arpa_filename);
    m_word_categories.read_category_gen_probs(cgenprobs_filename);
    m_word_categories.read_category_mem_probs(cmemprobs_filename);
    m_unk_root_node = unk_root_node;
    m_indexmap = get_class_index_map(m_word_categories.num_categories(), m_ln_arpa_model);
    m_history = nullptr;
    m_max_tokens = max_tokens;
    m_beam = beam;
    start_sentence();
}

CategoryNgram::~CategoryNgram()
{
    if (m_history) delete m_history;
}

bool
CategoryNgram::word_in_vocabulary(std::string word)
{
    if (word==UNK_SYMBOL) return false;
    if (word==CAP_UNK_SYMBOL) return false;
    auto cgenit = m_word_categories.m_category_gen_probs.find(word);
    auto cmemit = m_word_categories.m_category_mem_probs.find(word);
    if (cgenit==m_word_categories.m_category_gen_probs.end() || cgenit->second.size()==0)
        return false;
    if (cmemit==m_word_categories.m_category_mem_probs.end() || cmemit->second.size()==0)
        return false;
    return true;
}

void
CategoryNgram::start_sentence()
{
    if (m_history) delete m_history;
    m_history = new CatPerplexity::CategoryHistory(m_ln_arpa_model);
}

double
CategoryNgram::likelihood(std::string word)
{
    double ln_log_prob = 0.0;
    static unsigned long int num_vocab_words;
    static unsigned long int num_oov_words;
    if (word_in_vocabulary(word)) {
        ln_log_prob =
            CatPerplexity::likelihood(
                    m_ln_arpa_model,
                    m_word_categories,
                    m_indexmap,
                    num_vocab_words, num_oov_words,
                    word, *m_history,
                    m_unk_root_node, m_max_tokens, m_beam);

    }
    else {
        CatPerplexity::likelihood(
                m_ln_arpa_model,
                m_word_categories,
                m_indexmap,
                num_vocab_words, num_oov_words,
                word, *m_history,
                m_unk_root_node, m_max_tokens, m_beam);
    }
    return ln_log_prob;
}

double
CategoryNgram::sentence_end_likelihood()
{
    static unsigned long int num_vocab_words;
    static unsigned long int num_oov_words;
    return CatPerplexity::likelihood(
            m_ln_arpa_model,
            m_word_categories,
            m_indexmap,
            num_vocab_words, num_oov_words,
            SENTENCE_END_SYMBOL, *m_history,
            m_unk_root_node, m_max_tokens, m_beam);

}
