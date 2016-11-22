#include <cmath>
#include <sstream>
#include <algorithm>
#include <queue>
#include <thread>
#include <cfloat>

#include "Categories.hh"

using namespace std;


bool descending_token_sort(Token *a, Token *b)
{
    return (a->m_lp > b->m_lp);
}

Categories::Categories(int num_classes)
{
    m_num_categories = num_classes;
}

Categories::Categories(std::string filename,
                       const map<string, int> &counts,
                       int top_word_categories)
{
    SimpleFileInput infile(filename);
    m_stats.clear();

    string line;
    m_num_categories = 0;
    set<string> words;
    while(infile.getline(line)) {
        stringstream ss(line);

        string word;
        ss >> word;
        words.insert(word);

        int cl;
        vector<int> curr_classes;
        while (ss >> cl) {
            m_num_categories = max(cl+1, m_num_categories);
            curr_classes.push_back(cl);
        }
        flt_type curr_count = 1.0;
        if (counts.find(word) != counts.end()) curr_count = double(counts.at(word));
        for (auto cit=curr_classes.begin(); cit != curr_classes.end(); ++cit)
            m_stats[word][*cit] = curr_count/double(curr_classes.size());
    }

    // Set an own class for the most common words
    if (top_word_categories > 0) {
        map<int, string> sorted_counts;
        for (auto wit=counts.begin(); wit != counts.end(); ++wit)
            sorted_counts[wit->second] = wit->first;
        int sci = 0;
        for (auto wit=sorted_counts.rbegin(); wit != sorted_counts.rend(); wit++) {
            m_stats[wit->second].clear();
            m_stats[wit->second][m_num_categories] = 1.0;
            m_num_categories++;
            if (++sci >= top_word_categories) break;
        }
    }

    // Keep words with no class information in the stats
    for (auto wit=words.begin(); wit != words.end(); ++wit) {
        if (m_stats.find(*wit) != m_stats.end()) continue;
        m_stats[*wit] = CategoryProbs();
    }

    estimate_model();
}

void
Categories::accumulate(std::string word, int c, flt_type weight)
{
    m_stats[word][c] += weight;
}

void
Categories::accumulate(Categories &acc)
{
    for (auto wit=acc.m_stats.begin(); wit != acc.m_stats.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit)
            m_stats[wit->first][clit->first] += clit->second;
}


void
Categories::estimate_model()
{
    m_category_gen_probs.clear();
    m_category_mem_probs.clear();
    vector<flt_type> class_totals(m_num_categories, 0.0);
    map<string, flt_type> word_totals;

    for (auto wit=m_stats.begin(); wit != m_stats.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit) {
            class_totals[clit->first] += clit->second;
            word_totals[wit->first] += clit->second;
        }

    for (auto clit=class_totals.begin(); clit!= class_totals.end(); ++clit)
        *clit = log(*clit);
    for (auto wit=word_totals.begin(); wit!= word_totals.end(); ++wit)
        wit->second = log(wit->second);

    for (auto wit=m_stats.begin(); wit != m_stats.end(); ++wit) {

        // Keep words without analyses in the vocabulary
        m_category_gen_probs[wit->first] = CategoryProbs();
        m_category_mem_probs[wit->first] = CategoryProbs();

        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit) {
            flt_type wlp = log(clit->second) - class_totals[clit->first];
            flt_type clp = log(clit->second) - word_totals[wit->first];
            if (wlp > LP_PRUNE_LIMIT && !std::isinf(wlp)
                && clp > LP_PRUNE_LIMIT && !std::isinf(clp))
            {
                m_category_gen_probs[wit->first][clit->first] = clp;
                m_category_mem_probs[wit->first][clit->first] = wlp;
            }
        }
    }

    m_stats.clear();
}


int
Categories::num_words() const
{
    if (m_category_mem_probs.size() > 0) return m_category_mem_probs.size();
    else return m_stats.size();
}

int
Categories::num_words_with_categories() const
{
    int words = 0;
    if (m_category_mem_probs.size() > 0) {
        for (auto wit = m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit)
            if (wit->second.size() > 0) words++;
    }
    else {
        for (auto wit = m_stats.begin(); wit != m_stats.end(); ++wit)
            if (wit->second.size() > 0) words++;
    }
    return words;
}

int
Categories::num_categories() const
{
    return m_num_categories;
}


int
Categories::num_observed_categories() const
{
    set<int> categories;
    for (auto wit=m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit)
            categories.insert(clit->first);
    return categories.size();
}


int
Categories::num_category_gen_probs() const
{
    int num_cat_gen_probs = 0;
    for (auto wit=m_category_gen_probs.begin(); wit != m_category_gen_probs.end(); ++wit)
        num_cat_gen_probs += wit->second.size();
    return num_cat_gen_probs;
}


int
Categories::num_category_mem_probs() const
{
    int num_cat_mem_probs = 0;
    for (auto wit=m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit)
        num_cat_mem_probs += wit->second.size();
    return num_cat_mem_probs;
}


int
Categories::num_stats() const
{
    int num_stats = 0;
    for (auto wit=m_stats.begin(); wit != m_stats.end(); ++wit)
        num_stats += wit->second.size();
    return num_stats;
}


void
Categories::get_words(set<string> &words,
                      bool get_unanalyzed)
{
    words.clear();
    for (auto wit=m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit)
        if (wit->second.size() > 0 || get_unanalyzed)
            words.insert(wit->first);
}


void
Categories::get_unanalyzed_words(set<string> &words)
{
    words.clear();
    for (auto wit=m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit)
        if (wit->second.size() == 0)
            words.insert(wit->first);
}

void
Categories::get_unanalyzed_words(map<string, flt_type> &words)
{
    words.clear();
    for (auto wit=m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit)
        if (wit->second.size() == 0)
            words.insert(make_pair(wit->first, 0.0));
}

flt_type
Categories::log_likelihood(int c, std::string word) const
{
    auto wit = m_category_mem_probs.find(word);
    if (wit == m_category_mem_probs.end()) return MIN_LOG_PROB;
    auto prit = wit->second.find(c);
    if (prit == wit->second.end()) return MIN_LOG_PROB;
    return prit->second;
}


flt_type
Categories::log_likelihood(int c, const CategoryProbs *wcp) const
{
    if (wcp == nullptr) return MIN_LOG_PROB;
    auto prit = wcp->find(c);
    if (prit == wcp->end()) return MIN_LOG_PROB;
    return prit->second;
}


const CategoryProbs*
Categories::get_category_mem_probs(std::string word) const
{
    auto wit = m_category_mem_probs.find(word);
    if (wit == m_category_mem_probs.end()) return nullptr;
    return &(wit->second);
}


const CategoryProbs*
Categories::get_category_gen_probs(std::string word) const
{
    auto wit = m_category_gen_probs.find(word);
    if (wit == m_category_gen_probs.end()) return nullptr;
    return &(wit->second);
}


void
Categories::get_all_category_mem_probs(vector<map<string, flt_type> > &word_probs) const
{
    word_probs.resize(num_categories());
    for (auto wit=m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit)
        for (auto pit=wit->second.begin(); pit != wit->second.end(); ++pit)
            word_probs[pit->first][wit->first] = pit->second;
}


bool
Categories::assert_category_gen_probs() const
{
    std::map<string, flt_type> word_totals;
    for (auto wit=m_category_gen_probs.begin(); wit != m_category_gen_probs.end(); ++wit)
        word_totals[wit->first] = MIN_LOG_PROB;

    for (auto wit=m_category_gen_probs.begin(); wit != m_category_gen_probs.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit)
            word_totals[wit->first] = add_log_domain_probs(word_totals[wit->first], clit->second);

    bool ok = true;
    for (auto wit=word_totals.begin(); wit != word_totals.end(); ++wit) {
        if (wit->second == MIN_LOG_PROB) continue;
        if (fabs(wit->second) > 0.00001) {
            cerr << "assert, word " << wit->first << ": " << wit->second << endl;
            ok = false;
        }
    }

    return ok;
}

bool
Categories::assert_category_mem_probs() const
{
    vector<flt_type> category_totals(m_num_categories, MIN_LOG_PROB);
    for (auto wit=m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit)
            category_totals[clit->first] = add_log_domain_probs(category_totals[clit->first], clit->second);

    bool ok = true;
    for (unsigned int cl=0; cl<category_totals.size(); cl++) {
        if (category_totals[cl] == MIN_LOG_PROB) continue;
        if (fabs(category_totals[cl]) > 0.000001) {
            cerr << "assert, category " << cl << ": " << category_totals[cl] << endl;
            ok = false;
        }
    }

    return ok;
}

void
Categories::write_category_gen_probs(string fname) const
{
    SimpleFileOutput wcf(fname);

    for (auto wit=m_category_gen_probs.begin(); wit != m_category_gen_probs.end(); ++wit) {
        wcf << wit->first << "\t";
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit) {
            if (clit != wit->second.begin()) wcf << " ";
            wcf << clit->first << " " << clit->second;
        }
        wcf << "\n";
    }

    wcf.close();
}

void
Categories::write_category_mem_probs(string fname) const
{
    SimpleFileOutput wcf(fname);

    for (auto wit=m_category_mem_probs.begin(); wit != m_category_mem_probs.end(); ++wit) {
        wcf << wit->first << "\t";
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit) {
            if (clit != wit->second.begin()) wcf << " ";
            wcf << clit->first << " " << clit->second;
        }
        wcf << "\n";
    }

    wcf.close();
}

void
Categories::read_category_gen_probs(string fname)
{
    SimpleFileInput wcf(fname);

    string line;
    int max_category = 0;
    while (wcf.getline(line)) {
        stringstream ss(line);
        string word;
        int cat;
        flt_type prob;
        ss >> word;
        // Keep words without categories in the model
        m_category_gen_probs[word] = CategoryProbs();
        while (ss >> cat) {
            ss >> prob;
            m_category_gen_probs[word][cat] = prob;
            max_category = max(max_category, cat);
        }
    }
    m_num_categories = max(m_num_categories, max_category+1);
}

void
Categories::read_category_mem_probs(string fname)
{
    SimpleFileInput wcf(fname);

    string line;
    int max_category = 0;
    while (wcf.getline(line)) {
        stringstream ss(line);
        string word;
        int cat;
        flt_type prob;
        ss >> word;
        // Keep words without categories in the model
        m_category_mem_probs[word] = CategoryProbs();
        while (ss >> cat) {
            ss >> prob;
            m_category_mem_probs[word][cat] = prob;
            max_category = max(max_category, cat);
        }
    }
    m_num_categories = max(m_num_categories, max_category+1);
}


inline flt_type
get_cat_gen_lp(Token *tok,
               int context_length)
{
    flt_type cat_gen_lp = tok->m_gen_lp;
    int tmp = 1;
    while (tok->m_prev_token != nullptr && tmp++ < context_length) {
        tok = tok->m_prev_token;
        cat_gen_lp += tok->m_gen_lp;
    }
    return cat_gen_lp;
}


void
segment_sent(const std::vector<std::string> &words,
             const Ngram &ngram,
             const vector<int> &indexmap,
             const Categories &categories,
             unsigned int num_tokens,
             unsigned int num_final_tokens,
             flt_type prob_beam,
             vector<vector<Token*> > &tokens,
             vector<Token*> &pointers,
             unsigned long int *num_vocab_words,
             unsigned long int *num_oov_words,
             unsigned long int *num_unpruned_tokens,
             unsigned long int *num_pruned_tokens)
{
    pointers.clear();
    tokens.clear();
    tokens.resize(words.size()+2);
    static double log10_to_ln = log(10.0);

    Token *initial_token = new Token();
    initial_token->m_cng_node = ngram.sentence_start_node;
    tokens[0].push_back(initial_token);
    pointers.push_back(initial_token);

    for (unsigned int i=0; i<words.size(); i++) {

        const CategoryProbs *cgp = categories.get_category_gen_probs(words[i]);
        const CategoryProbs *cmp = categories.get_category_mem_probs(words[i]);

        if (cmp != nullptr && cmp->size() > 0) {
            if (num_vocab_words != nullptr) (*num_vocab_words)++;
        }
        else {
            if (num_oov_words != nullptr) (*num_oov_words)++;
        }

        vector<Token*> &curr_tokens = tokens[i];
        flt_type best_score = -FLT_MAX;
        flt_type worst_score = FLT_MAX;
        for (auto tit = curr_tokens.begin(); tit != curr_tokens.end(); ++tit) {

            Token &tok = *(*tit);

            flt_type cat_gen_lp = get_cat_gen_lp(&tok, ngram.max_order-1);

            // Categories are defined, iterate over memberships
            if (cmp != nullptr && cmp->size() > 0) {
                for (auto cit = cmp->cbegin(); cit != cmp->cend(); ++cit) {
                    int c = cit->first;

                    flt_type curr_score = tok.m_lp + cat_gen_lp;
                    flt_type ngram_lp = 0.0;
                    int ngram_node_idx = ngram.score(tok.m_cng_node, indexmap[c], ngram_lp);
                    curr_score += ngram_lp * log10_to_ln;
                    curr_score += cit->second;

                    if ((curr_score+prob_beam) < best_score) {
                        if (num_pruned_tokens != nullptr) (*num_pruned_tokens)++;
                        continue;
                    }
                    if (num_unpruned_tokens != nullptr) (*num_unpruned_tokens)++;
                    best_score = max(best_score, curr_score);
                    worst_score = min(worst_score, curr_score);

                    Token* new_tok = new Token(tok, c);
                    new_tok->m_lp = curr_score;
                    new_tok->m_gen_lp = cgp->at(cit->first);
                    new_tok->m_cng_node = ngram_node_idx;
                    tokens[i+1].push_back(new_tok);
                    pointers.push_back(new_tok);
                }
            }
            // Advance with the unk symbol
            else {
                Token* new_tok = new Token(tok, -1);
                new_tok->m_lp = tok.m_lp;
                new_tok->m_cng_node = ngram.advance(tok.m_cng_node, ngram.unk_symbol_idx);
                tokens[i+1].push_back(new_tok);
                pointers.push_back(new_tok);
            }

            // No classes defined, handles initial pass for words without a class
            // FIXME: implementation required
            /*
            else {
                for (auto ngramit = ctxt->cbegin(); ngramit != ctxt->cend(); ++ngramit) {
                    int c = ngramit->first;
                    // UNK mapping not allowed in this case
                    if (c == UNK_CLASS) continue;
                    if (c == START_CLASS) continue;

                    flt_type curr_score = tok.m_score;
                    curr_score += class_gen_score;
                    curr_score += ngramit->second;
                    // Heuristic p(w|c)=1 for all new words

                    if ((curr_score+prob_beam) < best_score) {
                        pruned++;
                        continue;
                    }
                    unpruned++;
                    best_score = max(best_score, curr_score);
                    worst_score = min(worst_score, curr_score);

                    Token* new_tok = new Token(tok, c);
                    new_tok->m_score = curr_score;
                    tokens[i].push_back(new_tok);
                    pointers.push_back(new_tok);
                }
            }
            */
        }

        if (i<words.size()-1)
            histogram_prune(tokens[i+1], num_tokens, worst_score, best_score);
        else
            histogram_prune(tokens[i+1], num_final_tokens, worst_score, best_score);
    }

    // Add sentence end scores
    vector<Token*> &curr_tokens = tokens[tokens.size()-2];
    for (auto tit = curr_tokens.begin(); tit != curr_tokens.end(); ++tit) {
        Token &tok = *(*tit);
        Token* new_tok = new Token(tok, -1);
        new_tok->m_lp = tok.m_lp + get_cat_gen_lp(&tok, ngram.max_order-1);
        flt_type ngram_lp = 0.0;
        new_tok->m_cng_node = ngram.score(tok.m_cng_node, ngram.sentence_end_symbol_idx, ngram_lp);
        new_tok->m_lp += ngram_lp * log10_to_ln;
        tokens.back().push_back(new_tok);
        pointers.push_back(new_tok);
    }
}


flt_type
collect_stats(const vector<string> &sent,
              const Ngram &ngram,
              const vector<int> &indexmap,
              const Categories &categories,
              Categories &stats,
              SimpleFileOutput *seqf,
              unsigned int num_tokens,
              unsigned int num_final_tokens,
              unsigned int num_parses,
              flt_type prob_beam,
              bool verbose,
              unsigned long int *num_vocab_words,
              unsigned long int *num_oov_words,
              unsigned long int *num_unpruned_tokens,
              unsigned long int *num_pruned_tokens)
{
    vector<vector<Token*> > tokens;
    vector<Token*> pointers;
    segment_sent(sent, ngram, indexmap, categories,
                 num_tokens, num_final_tokens, prob_beam,
                 tokens, pointers,
                 num_vocab_words, num_oov_words,
                 num_unpruned_tokens, num_pruned_tokens);

    vector<Token*> &final_tokens = tokens.back();

    if (final_tokens.size() == 0) {
        cerr << "No tokens in the final node, skipping sentence" << endl;
        for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
            delete *pit;
        return 0.0;
    }

    flt_type total_lp = MIN_LOG_PROB;
    for (auto tit = final_tokens.begin(); tit != final_tokens.end(); ++tit) {
        Token *tok = *tit;
        total_lp = add_log_domain_probs(total_lp, tok->m_lp);
    }

    if (std::isinf(total_lp) || std::isnan(total_lp)) {
        cerr << "Error, invalid total ll" << endl;
        for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
            delete *pit;
        return 0.0;
    }

    sort(final_tokens.begin(), final_tokens.end(), descending_token_sort);
    for (unsigned int i=0; i<final_tokens.size(); i++) {
        Token *tok = final_tokens[i];
        flt_type lp = std::min((flt_type)0.0, tok->m_lp-total_lp);
        vector<int> catseq; catseq.push_back(tok->m_category);
        while (tok->m_prev_token != nullptr) {
            tok = tok->m_prev_token;
            catseq.push_back(tok->m_category);
        }
        std::reverse(catseq.begin(), catseq.end());

        flt_type weight = exp(lp);
        for (unsigned int c=1; c<catseq.size()-1; c++) {
            if (catseq[c] == -1) continue; // FIXME skip unks
            stats.accumulate(sent[c-1], catseq[c], weight);
        }

        if (seqf != nullptr) {
            if (i<num_parses) {
                if (num_parses > 1) *seqf << weight << " ";
                *seqf << "<s>";
                for (unsigned int c=1; c<catseq.size()-1; c++) {
                    if (catseq[c] == -1) *seqf << " <unk>";
                    else *seqf << " " << catseq[c];
                }
                *seqf << " </s>\n";
            }
        }
    }

    for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
        delete *pit;

    return total_lp;
}


bool descending_int_flt_sort(const pair<int, flt_type> &i,
                             const pair<int, flt_type> &j)
{
    return (i.second > j.second);
}


void limit_num_categories(map<string, CategoryProbs> &probs,
                          int num_categories)
{
    for (auto wit=probs.begin(); wit != probs.end(); ++wit) {
        vector<pair<int, flt_type> > wprobs;
        for (auto pit=wit->second.begin(); pit != wit->second.end(); ++pit)
            wprobs.push_back(*pit);
        sort(wprobs.begin(), wprobs.end(), descending_int_flt_sort);
        wit->second.clear();
        for (int i=0; i<num_categories && i<(int)wprobs.size(); i++)
            wit->second.insert(wprobs[i]);
    }
}


void histogram_prune(vector<Token*> &tokens,
                     int num_tokens,
                     flt_type worst_score,
                     flt_type best_score)
{
    if ((int)tokens.size() <= num_tokens) return;

    int NUM_BINS = 100;
    flt_type range = best_score-worst_score;
    if (range == 0.0) {
        // Handle some special cases where histogram pruning fails
        // May happen for instance in the first training iterations
        if ((int)tokens.size() > (2*num_tokens)) tokens.resize(2*num_tokens);
        return;
    }

    vector<int> token_bins(tokens.size());
    vector<int> bin_counts(NUM_BINS, 0);
    for (int i=0; i<(int)tokens.size(); i++) {
        int bin = round((NUM_BINS-1) * ((best_score-tokens[i]->m_lp) / range));
        token_bins[i] = bin;
        bin_counts[bin]++;
    }

    int bin_limit = 0;
    int bin_token_count = 0;
    for (int i=0; i<NUM_BINS; i++) {
        bin_token_count += bin_counts[i];
        bin_limit = i;
        if (bin_token_count >= num_tokens) break;
    }

    vector<Token*> pruned_tokens;
    for (int i=0; i<(int)tokens.size(); i++)
        if (token_bins[i] <= bin_limit)
            pruned_tokens.push_back(tokens[i]);

    // Handle some special cases where histogram pruning fails
    // May happen for instance in the first training iterations
    if ((int)pruned_tokens.size() > (2*num_tokens)) {
        sort(pruned_tokens.begin(), pruned_tokens.end(), descending_token_sort);
        pruned_tokens.resize(2*num_tokens);
    }

    tokens.swap(pruned_tokens);
}

