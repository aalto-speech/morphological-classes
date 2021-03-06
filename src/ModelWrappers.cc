#include "ModelWrappers.hh"
#include "str.hh"

using namespace std;


void
LanguageModel::evaluate(
        string corpus_filename,
        string *probs_filename,
        int *ppl_num_words)
{
    SimpleFileOutput *prob_file = nullptr;
    if (probs_filename != nullptr)
        prob_file = new SimpleFileOutput(*probs_filename);

    SimpleFileInput infile(corpus_filename);
    string line;
    long int num_words = 0;
    long int num_sents = 0;
    long int num_oovs = 0;
    double total_ll = 0.0;
    long int empty_lines_count = 0;
    while (infile.getline(line)) {

        line = str::cleaned(line);
        if (line.length()==0) empty_lines_count++;

        stringstream ss(line);
        vector<string> words;
        string word;
        while (ss >> word) {
            if (word==SENTENCE_BEGIN_SYMBOL) continue;
            //if (word==SENTENCE_END_SYMBOL) continue;
            words.push_back(word);
        }
        if (words.size() == 0 || words.back() != SENTENCE_END_SYMBOL) words.push_back(SENTENCE_END_SYMBOL);

        double sent_ll = 0.0;
        this->start_sentence();
        for (auto wit = words.begin(); wit!=words.end(); ++wit) {
            if (prob_file && wit != words.begin()) *prob_file << " ";
            if (this->word_in_vocabulary(*wit)) {
                double word_ll = this->likelihood(*wit);
                sent_ll += word_ll;
                if (prob_file) *prob_file << word_ll;
                num_words++;
            }
            else {
                this->likelihood(UNK_SYMBOL);
                if (prob_file) *prob_file << "<unk>";
                num_oovs++;
            }
        }
        if (prob_file) *prob_file << "\n";

        total_ll += sent_ll;
        num_sents++;
    }

    cerr << endl;
    if (empty_lines_count > 0)
        cerr << "Warning, the file contained " << empty_lines_count << " empty lines" << endl;
    cerr << "Number of sentences: " << num_sents << endl;
    cerr << "Number of in-vocabulary words exluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oovs << endl;
    cerr << "OOV rate: " << double(num_oovs) / (double(num_oovs)+double(num_words-num_sents)) * 100.0 << " %" << endl;
    cerr << "Total log likelihood (ln): " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/2.302585092994046 << endl;

    double ppl = exp(-1.0/double(num_words) * total_ll);
    cerr << "Perplexity: " << ppl << endl;

    if (ppl_num_words != nullptr) {
        double wnppl = exp(-1.0/double(*ppl_num_words) * total_ll);
        cerr << "Word-normalized perplexity: " << wnppl << endl;
    }

    if (prob_file != nullptr) delete prob_file;
}


WordNgram::WordNgram(
        string arpa_filename,
        bool unk_root_node)
{
    m_ln_arpa_model.read_arpa(arpa_filename);
    m_unk_root_node = unk_root_node;
    start_sentence();
}

bool
WordNgram::word_in_vocabulary(string word)
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
WordNgram::likelihood(string word)
{
    double ln_log_prob = 0.0;
    if (word==SENTENCE_END_SYMBOL) {
        ln_log_prob = sentence_end_likelihood();
        start_sentence();
    }
    else if (word_in_vocabulary(word)) {
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
    double ln_log_prob = 0.0;
    m_current_node_id = m_ln_arpa_model.score(m_current_node_id, m_ln_arpa_model.sentence_end_symbol_idx, ln_log_prob);
    return ln_log_prob;
}

ClassNgram::ClassNgram(
        string arpa_filename,
        string cmemprobs_filename,
        bool unk_root_node)
{
    m_ln_arpa_model.read_arpa(arpa_filename);
    m_unk_root_node = unk_root_node;
    m_num_classes = read_class_memberships(cmemprobs_filename, m_class_memberships);
    m_indexmap = get_class_index_map(m_num_classes, m_ln_arpa_model);
    start_sentence();
}

bool
ClassNgram::word_in_vocabulary(string word)
{
    if (word==UNK_SYMBOL) return false;
    if (word==CAP_UNK_SYMBOL) return false;
    if (word==SENTENCE_END_SYMBOL) return true;
    return m_class_memberships.find(word)!=m_class_memberships.end();
}

void
ClassNgram::start_sentence()
{
    m_current_node_id = m_ln_arpa_model.sentence_start_node;
}

double
ClassNgram::likelihood(string word)
{
    double ln_log_prob = 0.0;
    if (word==SENTENCE_END_SYMBOL) {
        ln_log_prob = sentence_end_likelihood();
        start_sentence();
    }
    else if (word_in_vocabulary(word)) {
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
        string arpa_filename,
        string cgenprobs_filename,
        string cmemprobs_filename,
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
CategoryNgram::word_in_vocabulary(string word)
{
    if (word==UNK_SYMBOL) return false;
    if (word==CAP_UNK_SYMBOL) return false;
    if (word==SENTENCE_END_SYMBOL) return true;
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
CategoryNgram::likelihood(string word)
{
    double ln_log_prob = 0.0;
    static unsigned long int num_vocab_words;
    static unsigned long int num_oov_words;
    if (word==SENTENCE_END_SYMBOL) {
        ln_log_prob = sentence_end_likelihood();
        start_sentence();
    }
    else if (word_in_vocabulary(word)) {
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


SubwordNgram::SubwordNgram(
        string arpa_filename,
        string word_segs_filename)
{
    m_ln_arpa_model.read_arpa(arpa_filename);
    read_word_segs(word_segs_filename, true);

    m_root_node = m_ln_arpa_model.root_node;
    m_sentence_start_node = m_ln_arpa_model.sentence_start_node;
    auto wb_symbol = m_ln_arpa_model.vocabulary_lookup.find("<w>");
    if (wb_symbol != m_ln_arpa_model.vocabulary_lookup.end()) {
        m_root_node = m_ln_arpa_model.advance(m_root_node, wb_symbol->second);
        m_sentence_start_node = m_ln_arpa_model.advance(m_sentence_start_node, wb_symbol->second);
    }

    start_sentence();
}

bool
SubwordNgram::word_in_vocabulary(string word)
{
    if (word==UNK_SYMBOL) return false;
    if (word==CAP_UNK_SYMBOL) return false;
    return ((word==SENTENCE_END_SYMBOL) || m_word_segs.find(word)!=m_word_segs.end());
}

void
SubwordNgram::start_sentence()
{
    m_current_node_id = m_sentence_start_node;
}

double
SubwordNgram::likelihood(string word)
{
    double ln_log_prob = 0.0;
    if (word==SENTENCE_END_SYMBOL) {
        ln_log_prob = sentence_end_likelihood();
        start_sentence();
    }
    else if (word_in_vocabulary(word)) {
        for (auto swit = m_word_segs.at(word).begin(); swit != m_word_segs.at(word).end(); ++swit)
            m_current_node_id = m_ln_arpa_model.score(m_current_node_id, *swit, ln_log_prob);
    }
    else {
        m_current_node_id = m_root_node;
    }
    return ln_log_prob;
}

double
SubwordNgram::sentence_end_likelihood()
{
    double ln_log_prob = 0.0;
    m_current_node_id = m_ln_arpa_model.score(m_current_node_id, m_ln_arpa_model.sentence_end_symbol_idx, ln_log_prob);
    return ln_log_prob;
}

void
SubwordNgram::read_word_segs(string word_segs_fname, bool only_sws)
{
    cerr << "Reading word segmentations: " << word_segs_fname << endl;
    ifstream segf(word_segs_fname);
    if (!segf) throw string("Problem opening word segmentations.");

    string line;
    auto wb_symbol = m_ln_arpa_model.vocabulary_lookup.find("<w>");
    while (getline(segf, line)) {
        if (line.length() == 0) continue;
        string word, subword, concatenated;
        vector<string> sw_tokens;
        stringstream ss(line);

        if (only_sws) {
            word = "";
            while (ss >> subword) {
                sw_tokens.push_back(subword);
                word += subword;
            }
        } else {
            ss >> word;
            while (ss >> subword) {
                sw_tokens.push_back(subword);
                concatenated += subword;
            }
            if (concatenated != word || sw_tokens.size() == 0)
                throw "Erroneous segmentation: " + concatenated;
        }

        vector<int> swids;
        bool word_ok = true;
        for (auto swit=sw_tokens.begin(); swit != sw_tokens.end(); ++swit) {
            auto swlit = m_ln_arpa_model.vocabulary_lookup.find(*swit);
            if (swlit==m_ln_arpa_model.vocabulary_lookup.end()) {
                cerr << "Skipping word: " << word << endl;
                cerr << "Subword " + *swit + " not found in the subword n-gram" << endl;
                word_ok = false;
            } else
                swids.push_back(swlit->second);
        }

        if (word_ok) {
            if (wb_symbol != m_ln_arpa_model.vocabulary_lookup.end())
                swids.push_back(wb_symbol->second);
            m_word_segs[word] = swids;
        }
    }
}


InterpolatedLM::InterpolatedLM(
        LanguageModel *lm1,
        LanguageModel *lm2,
        double interpolation_weight) :
        m_lm1(lm1), m_lm2(lm2), m_interpolation_weight(interpolation_weight)
{
    if (interpolation_weight<0.0 || interpolation_weight>1.0) {
        cerr << "Invalid interpolation weight: " << interpolation_weight << endl;
        exit(EXIT_FAILURE);
    }
    m_first_log_iw = log(interpolation_weight);
    m_second_log_iw = log(1.0-interpolation_weight);
    start_sentence();
}

bool
InterpolatedLM::word_in_vocabulary(string word)
{
    return m_lm1->word_in_vocabulary(word) && m_lm2->word_in_vocabulary(word);
}

void
InterpolatedLM::start_sentence()
{
    m_lm1->start_sentence();
    m_lm2->start_sentence();
}

double
InterpolatedLM::likelihood(string word)
{
    return add_log_domain_probs(
            m_first_log_iw + m_lm1->likelihood(word),
            m_second_log_iw + m_lm2->likelihood(word));
}

double
InterpolatedLM::sentence_end_likelihood()
{
    return add_log_domain_probs(
            m_first_log_iw + m_lm1->sentence_end_likelihood(),
            m_second_log_iw + m_lm2->sentence_end_likelihood());
}
