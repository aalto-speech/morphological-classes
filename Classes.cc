#include <sstream>
#include <cmath>
#include <algorithm>
#include <queue>
#include <thread>
#include <cfloat>

#include "Classes.hh"

using namespace std;

#define MIN_NGRAM_PROB -20.0


WordClasses::WordClasses(int num_classes)
{
    m_num_classes = num_classes;
}

WordClasses::WordClasses(std::string filename,
                         const map<string, int> &counts,
                         int top_word_classes)
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
    if (top_word_classes > 0) {
        map<int, string> sorted_counts;
        for (auto wit=counts.begin(); wit != counts.end(); ++wit)
            sorted_counts[wit->second] = wit->first;
        int sci = 0;
        for (auto wit=sorted_counts.rbegin(); wit != sorted_counts.rend(); wit++) {
            m_stats[wit->second].clear();
            m_stats[wit->second][m_num_classes] = 1.0;
            m_num_classes++;
            if (++sci >= top_word_classes) break;
        }
    }

    // Keep words with no class information in the stats
    for (auto wit=words.begin(); wit != words.end(); ++wit) {
        if (m_stats.find(*wit) != m_stats.end()) continue;
        m_stats[*wit] = WordClassProbs();
    }

    estimate_model();
}

void
WordClasses::accumulate(std::string word, int c, flt_type weight)
{
    m_stats[word][c] += weight;
}

void
WordClasses::accumulate(WordClasses &acc)
{
    for (auto wit=acc.m_stats.begin(); wit != acc.m_stats.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit)
            m_stats[wit->first][clit->first] += clit->second;
}


void
WordClasses::estimate_model()
{
    m_class_gen_probs.clear();
    m_class_memberships.clear();
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
        m_class_gen_probs[wit->first] = WordClassProbs();
        m_class_memberships[wit->first] = WordClassProbs();

        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit) {
            flt_type wlp = log(clit->second) - class_totals[clit->first];
            flt_type clp = log(clit->second) - word_totals[wit->first];
            if (wlp > LP_PRUNE_LIMIT && !std::isinf(wlp)
                && clp > LP_PRUNE_LIMIT && !std::isinf(clp))
            {
                m_class_gen_probs[wit->first][clit->first] = clp;
                m_class_memberships[wit->first][clit->first] = wlp;
            }
        }
    }

    m_stats.clear();
}


int
WordClasses::num_words() const
{
    if (m_class_memberships.size() > 0) return m_class_memberships.size();
    else return m_stats.size();
}

int
WordClasses::num_words_with_classes() const
{
    int words = 0;
    if (m_class_memberships.size() > 0) {
        for (auto wit = m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit)
            if (wit->second.size() > 0) words++;
    }
    else {
        for (auto wit = m_stats.begin(); wit != m_stats.end(); ++wit)
            if (wit->second.size() > 0) words++;
    }
    return words;
}

int
WordClasses::num_classes() const
{
    return m_num_classes;
}


int
WordClasses::num_observed_classes() const
{
    set<int> classes;
    for (auto wit=m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit)
        for (auto clit = wit->second.begin(); clit != wit->second.end(); ++clit)
            classes.insert(clit->first);
    return classes.size();
}


int
WordClasses::num_class_probs() const
{
    int num_class_probs = 0;
    for (auto wit=m_class_gen_probs.begin(); wit != m_class_gen_probs.end(); ++wit)
        num_class_probs += wit->second.size();
    return num_class_probs;
}


int
WordClasses::num_word_probs() const
{
    int num_word_probs = 0;
    for (auto wit=m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit)
        num_word_probs += wit->second.size();
    return num_word_probs;
}


int
WordClasses::num_stats() const
{
    int num_stats = 0;
    for (auto wit=m_stats.begin(); wit != m_stats.end(); ++wit)
        num_stats += wit->second.size();
    return num_stats;
}


void
WordClasses::get_words(set<string> &words,
                       bool get_unanalyzed)
{
    words.clear();
    for (auto wit=m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit)
        if (wit->second.size() > 0 || get_unanalyzed)
            words.insert(wit->first);
}


void
WordClasses::get_unanalyzed_words(set<string> &words)
{
    words.clear();
    for (auto wit=m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit)
        if (wit->second.size() == 0)
            words.insert(wit->first);
}

void
WordClasses::get_unanalyzed_words(map<string, flt_type> &words)
{
    words.clear();
    for (auto wit=m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit)
        if (wit->second.size() == 0)
            words.insert(make_pair(wit->first, 0.0));
}

flt_type
WordClasses::log_likelihood(int c, std::string word) const
{
    auto wit = m_class_memberships.find(word);
    if (wit == m_class_memberships.end()) return MIN_LOG_PROB;
    auto prit = wit->second.find(c);
    if (prit == wit->second.end()) return MIN_LOG_PROB;
    return prit->second;
}


flt_type
WordClasses::log_likelihood(int c, const WordClassProbs *wcp) const
{
    if (wcp == nullptr) return MIN_LOG_PROB;
    auto prit = wcp->find(c);
    if (prit == wcp->end()) return MIN_LOG_PROB;
    return prit->second;
}


const WordClassProbs*
WordClasses::get_word_probs(std::string word) const
{
    auto wit = m_class_memberships.find(word);
    if (wit == m_class_memberships.end()) return nullptr;
    return &(wit->second);
}


const WordClassProbs*
WordClasses::get_class_probs(std::string word) const
{
    auto wit = m_class_gen_probs.find(word);
    if (wit == m_class_gen_probs.end()) return nullptr;
    return &(wit->second);
}


void
WordClasses::get_all_word_probs(vector<map<string, flt_type> > &word_probs) const
{
    word_probs.resize(num_classes());
    for (auto wit=m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit)
        for (auto pit=wit->second.begin(); pit != wit->second.end(); ++pit)
            word_probs[pit->first][wit->first] = pit->second;
}


bool
WordClasses::assert_class_probs() const
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
WordClasses::assert_word_probs() const
{
    vector<flt_type> class_totals(m_num_classes, MIN_LOG_PROB);
    for (auto wit=m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit)
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
WordClasses::write_class_probs(string fname) const
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
WordClasses::write_word_probs(string fname) const
{
    SimpleFileOutput wcf(fname);

    for (auto wit=m_class_memberships.begin(); wit != m_class_memberships.end(); ++wit) {
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
WordClasses::read_class_probs(string fname)
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
        m_class_gen_probs[word] = WordClassProbs();
        while (ss >> clss) {
            ss >> prob;
            m_class_gen_probs[word][clss] = prob;
            max_class = max(max_class, clss);
        }
    }
    m_num_classes = max(m_num_classes, max_class+1);
}

void
WordClasses::read_word_probs(string fname)
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
        m_class_memberships[word] = WordClassProbs();
        while (ss >> clss) {
            ss >> prob;
            m_class_memberships[word][clss] = prob;
            max_class = max(max_class, clss);
        }
    }
    m_num_classes = max(m_num_classes, max_class+1);
}


Unigram::Unigram(int num_classes)
{
    flt_type ll = log(1.0/double(num_classes));
    for (int i=0; i<num_classes; i++)
        m_unigrams[i] = ll;
}

const NgramCtxt*
Unigram::get_context(int c2, int c1) const
{
    return &m_unigrams;
}

flt_type
Unigram::log_likelihood(const NgramCtxt *ctxt, int c) const
{
    if (ctxt == nullptr) return MIN_NGRAM_PROB;
    auto ll = ctxt->find(c);
    if (ll == ctxt->end()) return MIN_NGRAM_PROB;
    return ll->second;
}

void
Unigram::accumulate(std::vector<int> &classes, flt_type weight)
{
    for (unsigned int i=2; i<classes.size(); i++)
        m_unigrams[classes[i]] += weight;
}

void
Unigram::accumulate(const ClassNgram *acc)
{
    const Unigram *ug = (Unigram*)acc;
    for (auto ugs=ug->m_unigrams.cbegin(); ugs!=ug->m_unigrams.cend(); ++ugs)
        m_unigrams[ugs->first] += ugs->second;
}

void
Unigram::estimate_model(bool discard_unks)
{
    if (discard_unks) m_unigrams.erase(UNK_CLASS);
    flt_type total_lp = MIN_LOG_PROB;
    for (auto ugit = m_unigrams.begin(); ugit != m_unigrams.end(); ++ugit) {
        ugit->second = log(ugit->second);
        total_lp = add_log_domain_probs(total_lp, ugit->second);
    }
    for (auto ugit = m_unigrams.begin(); ugit != m_unigrams.end(); ) {
        ugit->second -= total_lp;
        if (ugit->second < LP_PRUNE_LIMIT || std::isinf(ugit->second)) ugit = m_unigrams.erase(ugit);
        else ++ugit;
    }
}

void
Unigram::write_model(string fname) const
{
    SimpleFileOutput ugf(fname);

    for (auto uit=m_unigrams.begin(); uit != m_unigrams.end(); ++uit)
        ugf << uit->first << "\t" << uit->second << "\n";

    ugf.close();
}

void
Unigram::read_model(string fname)
{
    SimpleFileInput ugf(fname);

    string line;
    while (ugf.getline(line)) {
        stringstream ss(line);
        int clss;
        flt_type prob;
        ss >> clss >> prob;
        m_unigrams[clss] = prob;
    }
}

bool
Unigram::assert_model()
{
    flt_type total_prob = MIN_LOG_PROB;
    for (auto ngit = m_unigrams.begin(); ngit != m_unigrams.end(); ngit++)
        total_prob = add_log_domain_probs(total_prob, ngit->second);
    if (fabs(total_prob) > 0.00001) {
        cerr << "unigram total prob: " << total_prob << endl;
        return false;
    }
    return true;
}

int
Unigram::num_grams() const
{
    return m_unigrams.size();
}

ClassNgram*
Unigram::get_new() const
{
    return new Unigram();
}


const NgramCtxt*
Bigram::get_context(int c2, int c1) const
{
    auto trit = m_bigrams.find(c1);
    if (trit == m_bigrams.end()) return nullptr;
    return &(trit->second);
}

flt_type
Bigram::log_likelihood(const NgramCtxt *ctxt, int c) const
{
    if (ctxt == nullptr) return MIN_NGRAM_PROB;
    auto ll = ctxt->find(c);
    if (ll == ctxt->end()) return MIN_NGRAM_PROB;
    return ll->second;
}

void
Bigram::accumulate(std::vector<int> &classes, flt_type weight)
{
    for (unsigned int i=2; i<classes.size(); i++)
        m_bigrams[classes[i-1]][classes[i]] += weight;
}

void
Bigram::accumulate(const ClassNgram *acc)
{
    const Bigram *bgs = (Bigram*)acc;
    for (auto c1=bgs->m_bigrams.cbegin(); c1!=bgs->m_bigrams.cend(); ++c1)
        for (auto cit=c1->second.begin(); cit != c1->second.end(); ++cit)
            m_bigrams[c1->first][cit->first] += cit->second;
}

void
Bigram::estimate_model(bool discard_unks)
{
    if (discard_unks) m_bigrams.erase(UNK_CLASS);
    for (auto bgit = m_bigrams.begin(); bgit != m_bigrams.end(); ++bgit) {
        if (discard_unks) bgit->second.erase(UNK_CLASS);
        flt_type total_lp = MIN_LOG_PROB;
        for (auto git = bgit->second.begin(); git != bgit->second.end(); ++git) {
            git->second = log(git->second);
            total_lp = add_log_domain_probs(total_lp, git->second);
        }
        for (auto git = bgit->second.begin(); git != bgit->second.end();) {
            git->second -= total_lp;
            if (git->second < LP_PRUNE_LIMIT || std::isinf(git->second)) git = bgit->second.erase(git);
            else ++git;
        }
    }
}

void
Bigram::write_model(string fname) const
{
    SimpleFileOutput bgf(fname);

    for (auto c1=m_bigrams.begin(); c1 != m_bigrams.end(); ++c1)
        for (auto bgit=c1->second.begin(); bgit != c1->second.end(); ++bgit)
            bgf << c1->first << " " << bgit->first << "\t" << bgit->second << "\n";

    bgf.close();
}

void
Bigram::read_model(string fname)
{
    SimpleFileInput ugf(fname);

    string line;
    while (ugf.getline(line)) {
        stringstream ss(line);
        int ctxt, clss;
        flt_type prob;
        ss >> ctxt >> clss >> prob;
        m_bigrams[ctxt][clss] = prob;
    }
}

bool
Bigram::assert_model()
{
    for (auto bgit = m_bigrams.begin(); bgit != m_bigrams.end(); ++bgit) {
        flt_type total_prob = MIN_LOG_PROB;
        for (auto ngit = bgit->second.begin(); ngit != bgit->second.end(); ngit++)
            total_prob = add_log_domain_probs(total_prob, ngit->second);
        if (fabs(total_prob) > 0.00001) {
            cerr << bgit->first << "\ttotal prob: " << total_prob << endl;
            return false;
        }
    }
    return true;
}

int
Bigram::num_grams() const
{
    int gram_count = 0;
    for (auto bgit = m_bigrams.begin(); bgit != m_bigrams.end(); ++bgit)
        gram_count += bgit->second.size();
    return gram_count;
}

ClassNgram*
Bigram::get_new() const
{
    return new Bigram();
}


const NgramCtxt*
Trigram::get_context(int c2, int c1) const
{
    auto trit = m_trigrams.find(c2);
    if (trit == m_trigrams.end()) return nullptr;
    auto bgit = trit->second.find(c1);
    if (bgit == trit->second.end()) return nullptr;
    return &(bgit->second);
}

flt_type
Trigram::log_likelihood(const NgramCtxt *ctxt, int c) const
{
    if (ctxt == nullptr) return MIN_NGRAM_PROB;
    auto ll = ctxt->find(c);
    if (ll == ctxt->end()) return MIN_NGRAM_PROB;
    return ll->second;
}

void
Trigram::accumulate(std::vector<int> &classes, flt_type weight)
{
    for (unsigned int i=2; i<classes.size(); i++)
        m_trigrams[classes[i-2]][classes[i-1]][classes[i]] += weight;
}

void
Trigram::accumulate(const ClassNgram *acc)
{
    const Trigram *tgs = (Trigram*)acc;
    for (auto c2=tgs->m_trigrams.cbegin(); c2!=tgs->m_trigrams.cend(); ++c2)
        for (auto c1=c2->second.cbegin(); c1 != c2->second.cend(); ++c1)
            for (auto cit=c1->second.begin(); cit != c1->second.end(); ++cit)
                m_trigrams[c2->first][c1->first][cit->first] += cit->second;
}

void
Trigram::estimate_model(bool discard_unks)
{
    if (discard_unks) m_trigrams.erase(UNK_CLASS);
    for (auto trit = m_trigrams.begin(); trit != m_trigrams.end(); ++trit) {
        if (discard_unks) trit->second.erase(UNK_CLASS);
        for (auto bgit = trit->second.begin(); bgit != trit->second.end(); ++bgit) {
            if (discard_unks) bgit->second.erase(UNK_CLASS);
            flt_type total_lp = MIN_LOG_PROB;
            for (auto git = bgit->second.begin(); git != bgit->second.end(); ++git) {
                git->second = log(git->second);
                total_lp = add_log_domain_probs(total_lp, git->second);
            }
            for (auto git = bgit->second.begin(); git != bgit->second.end();) {
                git->second -= total_lp;
                if (git->second < LP_PRUNE_LIMIT || std::isinf(git->second)) git = bgit->second.erase(git);
                else ++git;
            }
        }
    }
}

void
Trigram::write_model(string fname) const
{
    SimpleFileOutput tgf(fname);

    for (auto c2=m_trigrams.begin(); c2 != m_trigrams.end(); ++c2)
        for (auto c1=c2->second.begin(); c1 != c2->second.end(); ++c1)
            for (auto tgit=c1->second.begin(); tgit != c1->second.end(); ++tgit)
                tgf << c2->first << " " << c1->first << " " << tgit->first << "\t" << tgit->second << "\n";

    tgf.close();
}

void
Trigram::read_model(string fname)
{
    SimpleFileInput ugf(fname);

    string line;
    while (ugf.getline(line)) {
        stringstream ss(line);
        int ctxt2, ctxt1, clss;
        flt_type prob;
        ss >> ctxt2 >> ctxt1 >> clss >> prob;
        m_trigrams[ctxt2][ctxt1][clss] = prob;
    }
}


bool
Trigram::assert_model()
{
    for (auto trit = m_trigrams.begin(); trit != m_trigrams.end(); ++trit)
        for (auto bgit = trit->second.begin(); bgit != trit->second.end(); ++bgit) {
            flt_type total_prob = MIN_LOG_PROB;
            for (auto ngit = bgit->second.begin(); ngit != bgit->second.end(); ngit++)
                total_prob = add_log_domain_probs(total_prob, ngit->second);
            if (fabs(total_prob) > 0.00001) {
                cerr << trit->first << " " << bgit->first
                     << "\ttotal prob: " << total_prob << endl;
                return false;
            }
        }
    return true;
}


int
Trigram::num_grams() const
{
    int gram_count = 0;
    for (auto trit = m_trigrams.begin(); trit != m_trigrams.end(); ++trit)
        for (auto bgit = trit->second.begin(); bgit != trit->second.end(); ++bgit)
            gram_count += bgit->second.size();
    return gram_count;
}

ClassNgram*
Trigram::get_new() const
{
    return new Trigram();
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
             const ClassNgram *ngram,
             const WordClasses *word_classes,
             flt_type prob_beam,
             unsigned int max_tokens,
             unsigned int max_final_tokens,
             unsigned long int &unpruned,
             unsigned long int &pruned,
             vector<vector<Token*> > &tokens,
             vector<Token*> &pointers)
{
    pointers.clear();
    tokens.clear();
    tokens.resize(words.size());

    Token *initial_token = new Token();
    tokens[0].push_back(initial_token);
    pointers.push_back(initial_token);

    Token *initial_token_2 = new Token();
    initial_token_2->m_prev_token = initial_token;
    tokens[1].push_back(initial_token_2);
    pointers.push_back(initial_token_2);

    for (unsigned int i=2; i<words.size(); i++) {

        const WordClassProbs *wcp = word_classes->get_word_probs(words[i]);
        const WordClassProbs *c2p = word_classes->get_class_probs(words[i-2]);
        const WordClassProbs *c1p = word_classes->get_class_probs(words[i-1]);

        vector<Token*> &curr_tokens = tokens[i-1];
        flt_type best_score = -FLT_MAX;
        flt_type worst_score = FLT_MAX;
        for (auto tit = curr_tokens.begin(); tit != curr_tokens.end(); ++tit) {

            Token &tok = *(*tit);

            const NgramCtxt *ctxt = ngram->get_context(tok.m_prev_token->m_class, tok.m_class);
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
                    curr_score += ngram->log_likelihood(ctxt, c);
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
        }

        if (i<words.size()-1)
            histogram_prune(tokens[i], max_tokens, worst_score, best_score);
        else
            histogram_prune(tokens[i], max_final_tokens, worst_score, best_score);
    }

}


flt_type
collect_stats(const vector<vector<string> > &sents,
              const ClassNgram *ngram,
              const WordClasses *word_classes,
              ClassNgram *ngram_stats,
              WordClasses *word_stats,
              unsigned int max_tokens,
              unsigned int max_final_tokens,
              unsigned int num_threads,
              unsigned int thread_index,
              flt_type *retval,
              int *skipped_sents,
              flt_type prob_beam,
              bool verbose)
{
    int sent_count = 0;
    int total_token_count = 0;
    flt_type total_ll = 0;
    if (skipped_sents != nullptr) *skipped_sents = 0;
    unsigned long int unpruned = 0;
    unsigned long int pruned = 0;

    for (unsigned int senti=0; senti<sents.size(); senti++) {

        if (verbose && senti > 0 && senti % 10000 == 0) {
            cerr << "thread " << thread_index << "\tline " << senti
                 << "\tn-grams: " << ngram_stats->num_grams()
                 << "\t(class,word) freqs: " << word_stats->num_stats() << endl;
            cerr << "pruning percentage: " << float(pruned)/float(pruned+unpruned) << endl;
        }

        if (num_threads > 0 && (senti % num_threads) != thread_index) continue;

        const vector<string> &words = sents[senti];

        vector<vector<Token*> > tokens;
        vector<Token*> pointers;
        segment_sent(words, ngram, word_classes,
                     prob_beam, max_tokens, max_final_tokens,
                     unpruned, pruned, tokens, pointers);

        vector<Token*> &final_tokens = tokens.back();

        if (final_tokens.size() == 0) {
            //cerr << "No tokens in final node, skipping sentence" << endl;
            if (skipped_sents != nullptr) (*skipped_sents)++;
            for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
                delete *pit;
            continue;
        }

        flt_type normalizer = MIN_LOG_PROB;
        for (auto tit = final_tokens.begin(); tit != final_tokens.end(); ++tit) {
            Token *tok = *tit;
            normalizer = add_log_domain_probs(normalizer, tok->m_score);
        }

        if (std::isinf(normalizer) || std::isnan(normalizer)) {
            cerr << "Nan or inf total ll, skipping sentence" << endl;
            for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
                delete *pit;
            continue;
        }

        for (auto tit = final_tokens.begin(); tit != final_tokens.end(); ++tit) {
            Token *tok = *tit;
            flt_type prob = tok->m_score - normalizer;
            if (prob > 0.0) prob = 0.0;
            vector<int> classes; classes.push_back(tok->m_class);
            while (tok->m_prev_token != nullptr) {
                tok = tok->m_prev_token;
                classes.push_back(tok->m_class);
            }
            std::reverse(classes.begin(), classes.end());

            flt_type weight = exp(prob);
            for (unsigned int i=1; i<classes.size(); i++)
                word_stats->accumulate(words[i], classes[i], weight);
            ngram_stats->accumulate(classes, weight);
        }

        sent_count++;
        total_token_count += final_tokens.size();
        total_ll += normalizer;

        for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
            delete *pit;
    }

    if (verbose && num_threads == 0)
        cerr << "Final tokens per sentence: " << float(total_token_count)/float(sent_count) << endl;
    if (retval != nullptr) *retval = total_ll;
    return total_ll;
}


flt_type
collect_stats_thr(const std::vector<std::vector<std::string> > &sents,
                  const ClassNgram *ngram,
                  const WordClasses *word_classes,
                  ClassNgram *ngram_stats,
                  WordClasses *word_stats,
                  unsigned int num_threads,
                  unsigned int max_tokens,
                  unsigned int max_final_tokens,
                  flt_type prob_beam,
                  bool verbose)
{
    vector<std::thread*> workers;
    vector<ClassNgram*> thr_ngram_stats(num_threads, nullptr);
    vector<WordClasses*> thr_word_stats(num_threads, nullptr);
    vector<flt_type> lls(num_threads, 0.0);
    vector<int> skipped_sents(num_threads, 0);

    for (unsigned int thri=0; thri<num_threads; thri++) {
        thr_ngram_stats[thri] = ngram_stats->get_new();
        thr_word_stats[thri] = new WordClasses(word_classes->num_classes());
        std::thread *thr = new std::thread(collect_stats,
                                           std::cref(sents),
                                           ngram,
                                           word_classes,
                                           thr_ngram_stats[thri],
                                           thr_word_stats[thri],
                                           max_tokens,
                                           max_final_tokens,
                                           num_threads,
                                           thri,
                                           &(lls[thri]),
                                           &(skipped_sents[thri]),
                                           prob_beam,
                                           verbose);
        workers.push_back(thr);
    }

    flt_type total_ll = 0.0;
    int skipped = 0;
    for (unsigned int thri=0; thri<num_threads; thri++) {
        workers[thri]->join();
        total_ll += lls[thri];
        skipped += skipped_sents[thri];
        ngram_stats->accumulate(thr_ngram_stats[thri]);
        word_stats->accumulate(*(thr_word_stats[thri]));
        delete thr_ngram_stats[thri];
        delete thr_word_stats[thri];
        delete workers[thri];
    }

    cerr << skipped << " training sentences were without valid probabilities" << endl;

    return total_ll;
}


bool descending_token_sort(Token *a, Token *b)
{
    return (a->m_score > b->m_score);
}


void
print_class_seqs(string &fname,
                 const vector<vector<string> > &sents,
                 const ClassNgram *ngram,
                 const WordClasses *word_classes,
                 unsigned int max_tokens,
                 flt_type prob_beam,
                 unsigned int max_parses)
{
    SimpleFileOutput seqf(fname);
    print_class_seqs(seqf, sents, ngram, word_classes,
                     max_tokens, prob_beam, max_parses);
    seqf.close();
}


void
print_class_seqs(SimpleFileOutput &seqf,
                 const vector<vector<string> > &sents,
                 const ClassNgram *ngram,
                 const WordClasses *word_classes,
                 unsigned int max_tokens,
                 flt_type prob_beam,
                 unsigned int max_parses)
{
    for (unsigned int senti=0; senti<sents.size(); senti++) {

        const vector<string> &words = sents[senti];

        unsigned long int unpruned;
        unsigned long int pruned;
        vector<vector<Token*> > tokens;
        vector<Token*> pointers;
        segment_sent(words, ngram, word_classes,
                     prob_beam, max_tokens, max_tokens,
                     unpruned, pruned, tokens, pointers);

        vector<Token*> &final_tokens = tokens.back();

        if (final_tokens.size() == 0) {
            cerr << "No tokens in final node, skipping sentence" << endl;
            for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
                delete *pit;
            continue;
        }

        sort(final_tokens.begin(), final_tokens.end(), descending_token_sort);
        final_tokens.resize(std::min(max_parses, (unsigned int)final_tokens.size()));

        flt_type normalizer = MIN_LOG_PROB;
        for (auto tit = final_tokens.begin(); tit != final_tokens.end(); ++tit)
            normalizer = add_log_domain_probs(normalizer, (*tit)->m_score);

        if (std::isinf(normalizer) || std::isnan(normalizer)) {
            cerr << "Nan or inf total ll, skipping sentence" << endl;
            for (auto pit = pointers.begin(); pit != pointers.end(); ++pit)
                delete *pit;
            continue;
        }

        flt_type best_prob = (*(final_tokens.begin()))->m_score - normalizer;
        for (auto tit = final_tokens.begin(); tit != final_tokens.end(); ++tit) {
            Token *tok = *tit;
            flt_type prob = tok->m_score - normalizer;
            if (best_prob - prob > prob_beam) break;
            if (prob > 0.0) prob = 0.0;
            vector<int> classes; classes.push_back(tok->m_class);
            while (tok->m_prev_token != nullptr) {
                tok = tok->m_prev_token;
                classes.push_back(tok->m_class);
            }
            std::reverse(classes.begin(), classes.end());

            flt_type weight = exp(prob);
            if (max_parses > 1) {
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
    }
}


bool descending_int_flt_sort(const pair<int, flt_type> &i,
                             const pair<int, flt_type> &j)
{
    return (i.second > j.second);
}


void limit_num_classes(map<string, WordClassProbs> &probs,
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

