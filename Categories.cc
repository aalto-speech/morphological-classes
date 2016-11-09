#include <sstream>
#include <cmath>
#include <algorithm>
#include <queue>
#include <thread>
#include <cfloat>

#include "Categories.hh"

using namespace std;

#define MIN_NGRAM_PROB -20.0


Categories::Categories(int num_classes)
{
    m_num_classes = num_classes;
}

Categories::Categories(std::string filename,
                       const map<string, int> &counts,
                       int top_word_categories)
{
    SimpleFileInput infile(filename);
    m_stats.clear();

    string line;
    m_num_classes = 0;
    set<string> words;
    while(infile.getline(line)) {
        stringstream ss(line);

        string word;
        ss >> word;
        words.insert(word);

        int cl;
        vector<int> curr_classes;
        while (ss >> cl) {
            m_num_classes = max(cl+1, m_num_classes);
            curr_classes.push_back(cl);
        }
        flt_type curr_count = 1.0;
        if (counts.find(word) != counts.end()) curr_count = double(counts.at(word));
        for (auto cit=curr_classes.begin(); cit != curr_classes.end(); ++cit)
            m_stats[word][*cit] = curr_count/double(curr_classes.size());
    }
    m_stats["<s>"][START_CLASS] = 1.0;
    m_stats["<unk>"][UNK_CLASS] = 1.0;

    // Set an own class for the most common words
    if (top_word_categories > 0) {
        map<int, string> sorted_counts;
        for (auto wit=counts.begin(); wit != counts.end(); ++wit)
            sorted_counts[wit->second] = wit->first;
        int sci = 0;
        for (auto wit=sorted_counts.rbegin(); wit != sorted_counts.rend(); wit++) {
            m_stats[wit->second].clear();
            m_stats[wit->second][m_num_classes] = 1.0;
            m_num_classes++;
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
    m_class_gen_probs.clear();
    m_class_mem_probs.clear();
    vector<flt_type> class_totals(m_num_classes, 0.0);
    map<string, flt_type> word_totals;

    // Keep class for the UNK symbol in the model even if no UNKs in training data
    if (m_stats.find("<unk>") == m_stats.end())
        m_stats["<unk>"][UNK_CLASS] = 1.0;

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
        m_class_gen_probs[wit->first] = CategoryProbs();
        m_class_mem_probs[wit->first] = CategoryProbs();

        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit) {
            flt_type wlp = log(clit->second) - class_totals[clit->first];
            flt_type clp = log(clit->second) - word_totals[wit->first];
            if (wlp > LP_PRUNE_LIMIT && !std::isinf(wlp)
                && clp > LP_PRUNE_LIMIT && !std::isinf(clp))
            {
                m_class_gen_probs[wit->first][clit->first] = clp;
                m_class_mem_probs[wit->first][clit->first] = wlp;
            }
        }
    }

    m_stats.clear();
}


int
Categories::num_words() const
{
    if (m_class_mem_probs.size() > 0) return m_class_mem_probs.size();
    else return m_stats.size();
}

int
Categories::num_words_with_categories() const
{
    int words = 0;
    if (m_class_mem_probs.size() > 0) {
        for (auto wit = m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit)
            if (wit->second.size() > 0) words++;
    }
    else {
        for (auto wit = m_stats.begin(); wit != m_stats.end(); ++wit)
            if (wit->second.size() > 0) words++;
    }
    return words;
}

int
Categories::num_classes() const
{
    return m_num_classes;
}


int
Categories::num_observed_categories() const
{
    set<int> classes;
    for (auto wit=m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit)
            classes.insert(clit->first);
    return classes.size();
}


int
Categories::num_category_gen_probs() const
{
    int num_class_probs = 0;
    for (auto wit=m_class_gen_probs.begin(); wit != m_class_gen_probs.end(); ++wit)
        num_class_probs += wit->second.size();
    return num_class_probs;
}


int
Categories::num_category_mem_probs() const
{
    int num_word_probs = 0;
    for (auto wit=m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit)
        num_word_probs += wit->second.size();
    return num_word_probs;
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
    for (auto wit=m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit)
        if (wit->second.size() > 0 || get_unanalyzed)
            words.insert(wit->first);
}


void
Categories::get_unanalyzed_words(set<string> &words)
{
    words.clear();
    for (auto wit=m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit)
        if (wit->second.size() == 0)
            words.insert(wit->first);
}

void
Categories::get_unanalyzed_words(map<string, flt_type> &words)
{
    words.clear();
    for (auto wit=m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit)
        if (wit->second.size() == 0)
            words.insert(make_pair(wit->first, 0.0));
}

flt_type
Categories::log_likelihood(int c, std::string word) const
{
    auto wit = m_class_mem_probs.find(word);
    if (wit == m_class_mem_probs.end()) return MIN_LOG_PROB;
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
    auto wit = m_class_mem_probs.find(word);
    if (wit == m_class_mem_probs.end()) return nullptr;
    return &(wit->second);
}


const CategoryProbs*
Categories::get_category_gen_probs(std::string word) const
{
    auto wit = m_class_gen_probs.find(word);
    if (wit == m_class_gen_probs.end()) return nullptr;
    return &(wit->second);
}


void
Categories::get_all_category_mem_probs(vector<map<string, flt_type> > &word_probs) const
{
    word_probs.resize(num_classes());
    for (auto wit=m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit)
        for (auto pit=wit->second.begin(); pit != wit->second.end(); ++pit)
            word_probs[pit->first][wit->first] = pit->second;
}


bool
Categories::assert_category_gen_probs() const
{
    std::map<string, flt_type> word_totals;
    for (auto wit=m_class_gen_probs.begin(); wit != m_class_gen_probs.end(); ++wit)
        word_totals[wit->first] = MIN_LOG_PROB;

    for (auto wit=m_class_gen_probs.begin(); wit != m_class_gen_probs.end(); ++wit)
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
    vector<flt_type> class_totals(m_num_classes, MIN_LOG_PROB);
    for (auto wit=m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit)
            class_totals[clit->first] = add_log_domain_probs(class_totals[clit->first], clit->second);

    bool ok = true;
    for (unsigned int cl=0; cl<class_totals.size(); cl++) {
        if (class_totals[cl] == MIN_LOG_PROB) continue;
        if (fabs(class_totals[cl]) > 0.000001) {
            cerr << "assert, class " << cl << ": " << class_totals[cl] << endl;
            ok = false;
        }
    }

    return ok;
}

void
Categories::write_category_gen_probs(string fname) const
{
    SimpleFileOutput wcf(fname);

    for (auto wit=m_class_gen_probs.begin(); wit != m_class_gen_probs.end(); ++wit) {
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

    for (auto wit=m_class_mem_probs.begin(); wit != m_class_mem_probs.end(); ++wit) {
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
    int max_class = 0;
    while (wcf.getline(line)) {
        stringstream ss(line);
        string word;
        int clss;
        flt_type prob;
        ss >> word;
        // Keep words without classes in the model
        m_class_gen_probs[word] = CategoryProbs();
        while (ss >> clss) {
            ss >> prob;
            m_class_gen_probs[word][clss] = prob;
            max_class = max(max_class, clss);
        }
    }
    m_num_classes = max(m_num_classes, max_class+1);
}

void
Categories::read_category_mem_probs(string fname)
{
    SimpleFileInput wcf(fname);

    string line;
    int max_class = 0;
    while (wcf.getline(line)) {
        stringstream ss(line);
        string word;
        int clss;
        flt_type prob;
        ss >> word;
        // Keep words without classes in the model
        m_class_mem_probs[word] = CategoryProbs();
        while (ss >> clss) {
            ss >> prob;
            m_class_mem_probs[word][clss] = prob;
            max_class = max(max_class, clss);
        }
    }
    m_num_classes = max(m_num_classes, max_class+1);
}


int
get_word_counts(string corpusfname,
                map<string, int> &counts)
{
    SimpleFileInput corpusf(corpusfname);

    int wc = 0;
    string line;
    while (corpusf.getline(line)) {
        stringstream ss(line);
        string word;
        while (ss >> word) {
            if (word == "<s>" || word == "</s>") continue;
            counts[word]++;
            wc++;
        }
        counts["<s>"]++;
        //counts["</s>"]++;
    }

    return wc;
}


int
read_sents(string corpusfname,
           vector<vector<string> > &sents,
           int maxlen,
           set<string> *vocab,
           int *num_word_tokens,
           int *num_word_types,
           int *num_unk_tokens,
           int *num_unk_types)
{
    SimpleFileInput corpusf(corpusfname);

    sents.clear();
    string line;
    set<string> unk_types;
    set<string> word_types;
    int unk_tokens = 0;
    int word_tokens = 0;

    while (corpusf.getline(line)) {
        stringstream ss(line);
        string word;
        vector<string> words = { "<s>", "<s>" };
        while (ss >> word) {
            if (word == "<s>" || word == "</s>") continue;
            words.push_back(word);
        }
        words.push_back("<s>");
        if ((int)words.size() > maxlen+3) continue;
        if ((int)words.size() == 3) continue;

        for (auto wit=words.begin(); wit != words.end(); ++wit) {
            if (*wit == "<s>") continue;
            if (vocab != nullptr && vocab->find(*wit) == vocab->end()) {
                unk_types.insert(*wit);
                unk_tokens++;
                wit->assign("<unk>");
            }
            else {
                word_types.insert(*wit);
                word_tokens++;
            }
        }

        sents.push_back(words);
    }

    if (num_word_tokens != nullptr) *num_word_tokens = word_tokens;
    if (num_word_types != nullptr) *num_word_types = word_types.size();
    if (num_unk_tokens != nullptr) *num_unk_tokens = unk_tokens;
    if (num_unk_types != nullptr) *num_unk_types = unk_types.size();

    return sents.size();
}


void
segment_sent(const std::vector<std::string> &words,
             const Ngram &ngram,
             const vector<int> &indexmap,
             const Categories &categories,
             unsigned int num_tokens,
             unsigned int num_final_tokens,
             flt_type prob_beam,
             unsigned long int &unpruned,
             unsigned long int &pruned,
             vector<vector<Token*> > &tokens,
             vector<Token*> &pointers)
{
    pointers.clear();
    tokens.clear();
    tokens.resize(words.size());

    Token *initial_token = new Token();
    initial_token->m_cng_node = ngram.sentence_start_node;
    tokens[0].push_back(initial_token);
    pointers.push_back(initial_token);

    for (unsigned int i=1; i<words.size(); i++) {

        const CategoryProbs *wcp = categories.get_category_mem_probs(words[i]);
        const CategoryProbs *c2p = categories.get_category_gen_probs(words[i-2]);
        const CategoryProbs *c1p = categories.get_category_gen_probs(words[i-1]);

        vector<Token*> &curr_tokens = tokens[i-1];
        flt_type best_score = -FLT_MAX;
        flt_type worst_score = FLT_MAX;
        for (auto tit = curr_tokens.begin(); tit != curr_tokens.end(); ++tit) {

            Token &tok = *(*tit);

            // FIXME: implementation required
            //const NgramCtxt *ctxt = ngram->get_context(tok.m_prev_token->m_class, tok.m_class);
            // Allow all n-grams with the MIN_NGRAM_PROB probability
            //if (ctxt == nullptr) continue;

            flt_type class_gen_score = 0.0;
            if (c2p != nullptr) {
                auto c2it = c2p->find(tok.m_prev_token->m_class);
                if (c2it != c2p->end()) class_gen_score += c2it->second;
            }
            if (c1p != nullptr) {
                auto c1it = c1p->find(tok.m_class);
                if (c1it != c1p->end()) class_gen_score += c1it->second;
            }

            // Classes are defined, iterate over class memberships
            // Ngram likelihood is MIN_NGRAM_PROB for n-grams not in the model
            if (wcp != nullptr && wcp->size() > 0) {
                for (auto cit = wcp->cbegin(); cit != wcp->cend(); ++cit) {
                    int c = cit->first;

                    flt_type curr_score = tok.m_score;
                    curr_score += class_gen_score;
                    //curr_score += ngram->log_likelihood(ctxt, c);
                    curr_score += cit->second;

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
            histogram_prune(tokens[i], num_tokens, worst_score, best_score);
        else
            histogram_prune(tokens[i], num_final_tokens, worst_score, best_score);
    }

}


flt_type
collect_stats(const vector<string> &sent,
              const Ngram &ngram,
              const vector<int> &indexmap,
              const Categories &categories,
              Categories &stats,
              SimpleFileOutput &seqf,
              unsigned int num_tokens,
              unsigned int num_final_tokens,
              unsigned int num_parses,
              flt_type prob_beam,
              bool verbose)
{
    unsigned long int unpruned = 0;
    unsigned long int pruned = 0;

    vector<vector<Token*> > tokens;
    vector<Token*> pointers;
    segment_sent(sent, ngram, indexmap, categories,
                 num_tokens, num_final_tokens, prob_beam,
                 unpruned, pruned, tokens, pointers);

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
        total_lp = add_log_domain_probs(total_lp, tok->m_score);
    }

    if (std::isinf(total_lp) || std::isnan(total_lp)) {
        cerr << "Error, invalid total ll" << endl;
        for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
            delete *pit;
        return 0.0;
    }

    for (auto tit = final_tokens.begin(); tit != final_tokens.end(); ++tit) {
        Token *tok = *tit;
        flt_type lp = std::min((flt_type)0.0, tok->m_score-total_lp);
        vector<int> classes; classes.push_back(tok->m_class);
        while (tok->m_prev_token != nullptr) {
            tok = tok->m_prev_token;
            classes.push_back(tok->m_class);
        }
        std::reverse(classes.begin(), classes.end());

        flt_type weight = exp(lp);
        for (unsigned int i=1; i<classes.size(); i++)
            stats.accumulate(sent[i], classes[i], weight);

        if (num_parses > 1) {
            seqf << weight;
            seqf << " ";
        }
        seqf << "<s>";
        for (unsigned int i=2; i<classes.size()-1; i++)
            seqf << " " << classes[i];
        seqf << " </s>\n";
    }

    for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
        delete *pit;

    return total_lp;
}


bool descending_token_sort(Token *a, Token *b)
{
    return (a->m_score > b->m_score);
}


bool descending_int_flt_sort(const pair<int, flt_type> &i,
                             const pair<int, flt_type> &j)
{
    return (i.second > j.second);
}


void limit_num_classes(map<string, CategoryProbs> &probs,
                       int num_classes)
{
    for (auto wit=probs.begin(); wit != probs.end(); ++wit) {
        vector<pair<int, flt_type> > wprobs;
        for (auto pit=wit->second.begin(); pit != wit->second.end(); ++pit)
            wprobs.push_back(*pit);
        sort(wprobs.begin(), wprobs.end(), descending_int_flt_sort);
        wit->second.clear();
        for (int i=0; i<num_classes && i<(int)wprobs.size(); i++)
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
        int bin = round((NUM_BINS-1) * ((best_score-tokens[i]->m_score) / range));
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

