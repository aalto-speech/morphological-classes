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
        :Exchanging()
{
}

Splitting::Splitting(int num_classes,
        const std::map<std::string, int>& word_classes,
        string corpus_fname)
        :Exchanging(num_classes, word_classes, corpus_fname)
{
}

void
Splitting::freq_split(const set<int>& words,
        set<int>& class1_words,
        set<int>& class2_words,
        vector<int>& vowords) const
{
    vowords.clear();
    multimap<int, int> ordered_words;
    for (auto wit = words.begin(); wit!=words.end(); ++wit)
        ordered_words.insert(make_pair(m_word_counts[*wit], *wit));
    bool class1_assign = true;
    for (auto owit = ordered_words.rbegin(); owit!=ordered_words.rend(); ++owit) {
        if (class1_assign) class1_words.insert(owit->second);
        else class2_words.insert(owit->second);
        class1_assign = !class1_assign;
        vowords.push_back(owit->second);
    }
}

int
Splitting::do_split(int class_idx,
        const set<int>& class1_words,
        const set<int>& class2_words)
{
    if (m_classes[class_idx].size()<2) {
        cerr << "Error, trying to split a class with " << m_classes[class_idx].size() << " words" << endl;
        exit(1);
    }

    int class2_idx = -1;
    for (int i = 0; i<(int) m_classes.size(); i++)
        if (m_classes[i].size()==0) {
            class2_idx = i;
            break;
        }
    if (class2_idx==-1) {
        class2_idx = m_classes.size();
        m_classes.resize(m_classes.size()+1);
        m_class_counts.resize(m_classes.size(), 0);
        m_class_bigram_counts.resize(m_classes.size());
        for (int i = 0; i<(int) m_class_bigram_counts.size(); i++)
            m_class_bigram_counts[i].resize(m_classes.size(), 0);
    }

    // Update class unigram counts
    int class1_count = 0, class2_count = 0;
    for (auto wit = class1_words.begin(); wit!=class1_words.end(); ++wit)
        class1_count += m_word_counts[*wit];
    for (auto wit = class2_words.begin(); wit!=class2_words.end(); ++wit)
        class2_count += m_word_counts[*wit];
    if (class1_count+class2_count!=m_class_counts[class_idx]) {
        cerr << "Error, class counts do not match with the unsplit count." << endl;
        exit(1);
    }
    m_class_counts[class_idx] = class1_count;
    m_class_counts[class2_idx] = class2_count;

    // Update class bigram counts
    for (int i = 0; i<(int) m_class_bigram_counts.size(); i++)
        m_class_bigram_counts[i][class_idx] = 0;
    for (int j = 0; j<(int) m_class_bigram_counts[class_idx].size(); j++)
        m_class_bigram_counts[class_idx][j] = 0;

    for (int i = 0; i<(int) m_class_word_counts.size(); i++) {
        auto cit = m_class_word_counts[i].find(class_idx);
        if (cit!=m_class_word_counts[i].end()) cit->second = 0;
    }
    for (int i = 0; i<(int) m_word_class_counts.size(); i++) {
        auto cit = m_word_class_counts[i].find(class_idx);
        if (cit!=m_word_class_counts[i].end()) cit->second = 0;
    }

    for (unsigned int i = 0; i<m_word_bigram_counts.size(); i++) {
        int src_class = m_word_classes[i];
        map<int, int>& curr_bigram_ctxt = m_word_bigram_counts[i];
        for (auto bgit = curr_bigram_ctxt.begin(); bgit!=curr_bigram_ctxt.end(); ++bgit) {
            int tgt_class = m_word_classes[bgit->first];
            if (src_class!=class_idx && tgt_class!=class_idx) continue;

            int new_src_class = src_class;
            if (class2_words.find(i)!=class2_words.end()) {
                new_src_class = class2_idx;
                m_class_word_counts[bgit->first][new_src_class] += bgit->second;
            }
            else if (class1_words.find(i)!=class1_words.end())
                m_class_word_counts[bgit->first][new_src_class] += bgit->second;

            int new_tgt_class = tgt_class;
            if (class2_words.find(bgit->first)!=class2_words.end()) {
                new_tgt_class = class2_idx;
                m_word_class_counts[i][new_tgt_class] += bgit->second;
            }
            else if (class1_words.find(bgit->first)!=class1_words.end())
                m_word_class_counts[i][new_tgt_class] += bgit->second;

            m_class_bigram_counts[new_src_class][new_tgt_class] += bgit->second;
        }
    }

    m_classes[class_idx] = class1_words;
    m_classes[class2_idx] = class2_words;
    for (auto wit = class2_words.begin(); wit!=class2_words.end(); ++wit)
        m_word_classes[*wit] = class2_idx;
    m_num_classes++;

    return class2_idx;
}

int
Splitting::iterate_exchange_local(int class1_idx,
        int class2_idx,
        vector<int>& owords,
        int num_iterations)
{
    int num_exchanges = 0;

    for (int i = 0; i<num_iterations; i++) {
        for (auto wit = owords.begin(); wit!=owords.end(); ++wit) {
            int curr_class = m_word_classes[*wit];
            assert(curr_class==class1_idx || curr_class==class2_idx);
            int tentative_class;
            if (curr_class==class1_idx) tentative_class = class2_idx;
            else tentative_class = class1_idx;
            double ll_diff = evaluate_exchange(*wit, curr_class, tentative_class);
            if (ll_diff>0.0) {
                do_exchange(*wit, curr_class, tentative_class);
                num_exchanges++;
            }
        }
    }

    return num_exchanges;
}

