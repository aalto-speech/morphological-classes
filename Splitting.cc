#include <algorithm>
#include <sstream>
#include <cassert>
#include <cmath>
#include <ctime>
#include <thread>

#include "Splitting.hh"
#include "io.hh"
#include "defs.hh"

using namespace std;


Splitting::Splitting()
    : Merging()
{
}


Splitting::Splitting(int num_classes,
                     const std::map<std::string, int> &word_classes,
                     string corpus_fname)
    : Merging(num_classes, word_classes, corpus_fname)
{
}


inline void
evaluate_ll_diff(double &ll_diff,
                 int old_count,
                 int new_count)
{
    if (old_count != 0)
        ll_diff -= old_count * log(old_count);
    if (new_count != 0)
        ll_diff += new_count * log(new_count);
}


inline int
get_count(const map<int, int> &ctxt,
                     int element)
{
    auto it = ctxt.find(element);
    if (it != ctxt.end()) return it->second;
    else return 0;
}


double
Splitting::evaluate_exchange(int word,
                             int curr_class,
                             int tentative_class) const
{
    double ll_diff = 0.0;
    int wc = m_word_counts[word];
    const map<int, int> &wb_ctxt = m_word_bigram_counts.at(word);
    const map<int, int> &cw_counts = m_class_word_counts.at(word);
    const map<int, int> &wc_counts = m_word_class_counts.at(word);

    ll_diff += 2 * (m_class_counts[curr_class]) * log(m_class_counts[curr_class]);
    ll_diff -= 2 * (m_class_counts[curr_class]-wc) * log(m_class_counts[curr_class]-wc);
    ll_diff += 2 * (m_class_counts[tentative_class]) * log(m_class_counts[tentative_class]);
    ll_diff -= 2 * (m_class_counts[tentative_class]+wc) * log(m_class_counts[tentative_class]+wc);

    for (auto wcit=wc_counts.begin(); wcit != wc_counts.end(); ++wcit) {
        if (wcit->first == curr_class) continue;
        if (wcit->first == tentative_class) continue;

        int curr_count = m_class_bigram_counts[curr_class][wcit->first];
        int new_count = curr_count - wcit->second;
        evaluate_ll_diff(ll_diff, curr_count, new_count);

        curr_count = m_class_bigram_counts[tentative_class][wcit->first];
        new_count = curr_count + wcit->second;
        evaluate_ll_diff(ll_diff, curr_count, new_count);
    }

    for (auto wcit=cw_counts.begin(); wcit != cw_counts.end(); ++wcit) {
        if (wcit->first == curr_class) continue;
        if (wcit->first == tentative_class) continue;

        int curr_count = m_class_bigram_counts[wcit->first][curr_class];
        int new_count = curr_count - wcit->second;
        evaluate_ll_diff(ll_diff, curr_count, new_count);

        curr_count = m_class_bigram_counts[wcit->first][tentative_class];
        new_count = curr_count + wcit->second;
        evaluate_ll_diff(ll_diff, curr_count, new_count);
    }

    int self_count = 0;
    auto scit = wb_ctxt.find(word);
    if (scit != wb_ctxt.end()) self_count = scit->second;

    int curr_count = m_class_bigram_counts[curr_class][tentative_class];
    int new_count = curr_count - get_count(wc_counts, tentative_class)
            + get_count(cw_counts, curr_class) - self_count;
    evaluate_ll_diff(ll_diff, curr_count, new_count);

    curr_count = m_class_bigram_counts[tentative_class][curr_class];
    new_count = curr_count - get_count(cw_counts, tentative_class)
            + get_count(wc_counts, curr_class) - self_count;
    evaluate_ll_diff(ll_diff, curr_count, new_count);

    curr_count = m_class_bigram_counts[curr_class][curr_class];
    new_count = curr_count - get_count(wc_counts, curr_class)
            - get_count(cw_counts, curr_class) + self_count;
    evaluate_ll_diff(ll_diff, curr_count, new_count);

    curr_count = m_class_bigram_counts[tentative_class][tentative_class];
    new_count = curr_count + get_count(wc_counts, tentative_class)
            + get_count(cw_counts, tentative_class) + self_count;
    evaluate_ll_diff(ll_diff, curr_count, new_count);

    return ll_diff;
}


void
Splitting::do_exchange(int word,
                      int prev_class,
                      int new_class)
{
    int wc = m_word_counts[word];
    m_class_counts[prev_class] -= wc;
    m_class_counts[new_class] += wc;

    map<int, int> &bctxt = m_word_bigram_counts[word];
    for (auto wit = bctxt.begin(); wit != bctxt.end(); ++wit) {
        if (wit->first == word) continue;
        int tgt_class = m_word_classes[wit->first];
        m_class_bigram_counts[prev_class][tgt_class] -= wit->second;
        m_class_bigram_counts[new_class][tgt_class] += wit->second;
        m_class_word_counts[wit->first][prev_class] -= wit->second;
        m_class_word_counts[wit->first][new_class] += wit->second;
    }

    map<int, int> &rbctxt = m_word_rev_bigram_counts[word];
    for (auto wit = rbctxt.begin(); wit != rbctxt.end(); ++wit) {
        if (wit->first == word) continue;
        int src_class = m_word_classes[wit->first];
        m_class_bigram_counts[src_class][prev_class] -= wit->second;
        m_class_bigram_counts[src_class][new_class] += wit->second;
        m_word_class_counts[wit->first][prev_class] -= wit->second;
        m_word_class_counts[wit->first][new_class] += wit->second;
    }

    auto wit = bctxt.find(word);
    if (wit != bctxt.end()) {
        m_class_bigram_counts[prev_class][prev_class] -= wit->second;
        m_class_bigram_counts[new_class][new_class] += wit->second;
        m_class_word_counts[word][prev_class] -= wit->second;
        m_class_word_counts[word][new_class] += wit->second;
        m_word_class_counts[word][prev_class] -= wit->second;
        m_word_class_counts[word][new_class] += wit->second;
    }

    m_classes[prev_class].erase(word);
    m_classes[new_class].insert(word);
    m_word_classes[word] = new_class;
}


void
Splitting::random_split(const set<int> &words,
                        set<int> &class1_words,
                        set<int> &class2_words) const
{
    vector<int> _words(words.begin(), words.end());
    std::random_shuffle(_words.begin(), _words.end());
    class1_words.clear(); class2_words.clear();
    class1_words.insert(_words.begin(), _words.begin() + _words.size()/2);
    class2_words.insert(_words.begin() + _words.size()/2, _words.end());
}


void
Splitting::freq_split(const set<int> &words,
                      set<int> &class1_words,
                      set<int> &class2_words,
                      vector<int> &vowords) const
{
    vowords.clear();
    multimap<int, int> ordered_words;
    for (auto wit=words.begin(); wit != words.end(); ++wit)
        ordered_words.insert(make_pair(m_word_counts[*wit], *wit));
    bool class1_assign = true;
    for (auto owit=ordered_words.rbegin(); owit != ordered_words.rend(); ++owit) {
        if (class1_assign) class1_words.insert(owit->second);
        else class2_words.insert(owit->second);
        class1_assign = !class1_assign;
        vowords.push_back(owit->second);
    }
}


int
Splitting::do_split(int class_idx,
                    const set<int> &class1_words,
                    const set<int> &class2_words)
{
    if (m_classes[class_idx].size() < 2) {
        cerr << "Error, trying to split a class with " << m_classes[class_idx].size() << " words" << endl;
        exit(1);
    }

    int class2_idx = -1;
    for (int i=0; i<(int)m_classes.size(); i++)
        if (m_classes[i].size() == 0) {
            class2_idx = i;
            break;
        }
    if (class2_idx == -1) {
        class2_idx = m_classes.size();
        m_classes.resize(m_classes.size()+1);
        m_class_counts.resize(m_classes.size(), 0);
        m_class_bigram_counts.resize(m_classes.size());
        m_class_bigram_counts.back().resize(m_classes.size(), 0);
        for (int i=0; i<(int)m_class_bigram_counts[i].size(); i++)
            m_class_bigram_counts[i].resize(m_classes.size(), 0);
    }

    // Update class unigram counts
    int class1_count = 0, class2_count = 0;
    for (auto wit=class1_words.begin(); wit != class1_words.end(); ++wit)
        class1_count += m_word_counts[*wit];
    for (auto wit=class2_words.begin(); wit != class2_words.end(); ++wit)
        class2_count += m_word_counts[*wit];
    if (class1_count+class2_count != m_class_counts[class_idx]) {
        cerr << "Error, class counts do not match with the unsplit count." << endl;
        exit(1);
    }
    m_class_counts[class_idx] = class1_count;
    m_class_counts[class2_idx] = class2_count;

    // Update class bigram counts
    for (int i=0; i<(int)m_class_bigram_counts.size(); i++)
        m_class_bigram_counts[i][class_idx] = 0;
    for (int j=0; j<(int)m_class_bigram_counts[class_idx].size(); j++)
        m_class_bigram_counts[class_idx][j] = 0;

    for (int i=0; i<(int)m_class_word_counts.size(); i++) {
        auto cit = m_class_word_counts[i].find(class_idx);
        if (cit != m_class_word_counts[i].end()) cit->second = 0;
    }
    for (int i=0; i<(int)m_word_class_counts.size(); i++) {
        auto cit = m_word_class_counts[i].find(class_idx);
        if (cit != m_word_class_counts[i].end()) cit->second = 0;
    }

    for (unsigned int i=0; i<m_word_bigram_counts.size(); i++) {
        int src_class = m_word_classes[i];
        map<int, int> &curr_bigram_ctxt = m_word_bigram_counts[i];
        for (auto bgit = curr_bigram_ctxt.begin(); bgit != curr_bigram_ctxt.end(); ++bgit) {
            int tgt_class = m_word_classes[bgit->first];
            if (src_class != class_idx && tgt_class != class_idx) continue;

            int new_src_class = src_class;
            if (class2_words.find(i) != class2_words.end()) {
                new_src_class = class2_idx;
                m_class_word_counts[bgit->first][new_src_class] += bgit->second;
            }
            else if (class1_words.find(i) != class1_words.end())
                m_class_word_counts[bgit->first][new_src_class] += bgit->second;

            int new_tgt_class = tgt_class;
            if (class2_words.find(bgit->first) != class2_words.end()) {
                new_tgt_class = class2_idx;
                m_word_class_counts[i][new_tgt_class] += bgit->second;
            }
            else if (class1_words.find(bgit->first) != class1_words.end())
                m_word_class_counts[i][new_tgt_class] += bgit->second;

            m_class_bigram_counts[new_src_class][new_tgt_class] += bgit->second;
        }
    }

    m_classes[class_idx] = class1_words;
    m_classes[class2_idx] = class2_words;
    for (auto wit=class2_words.begin(); wit != class2_words.end(); ++wit)
        m_word_classes[*wit] = class2_idx;
    m_num_classes++;

    return class2_idx;
}


int
Splitting::iterate_exchange_local(int class1_idx,
                                  int class2_idx,
                                  vector<int> &owords,
                                  int num_iterations)
{
    int num_exchanges = 0;

    for (int i=0; i<num_iterations; i++) {
        for (auto wit=owords.begin(); wit != owords.end(); ++wit) {
            int curr_class = m_word_classes[*wit];
            assert(curr_class == class1_idx || curr_class == class2_idx);
            int tentative_class;
            if (curr_class == class1_idx) tentative_class = class2_idx;
            else tentative_class = class1_idx;
            double ll_diff = evaluate_exchange(*wit, curr_class, tentative_class);
            if (ll_diff > 0.0) {
                do_exchange(*wit, curr_class, tentative_class);
                num_exchanges++;
            }
        }
    }

    return num_exchanges;
}

