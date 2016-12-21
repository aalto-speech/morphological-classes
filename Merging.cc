#include <algorithm>
#include <sstream>
#include <cmath>
#include <ctime>
#include <cassert>
#include <thread>
#include <functional>
#include <iterator>
#include <algorithm>

#include "ExchangeAlgorithm.hh"
#include "io.hh"
#include "defs.hh"

using namespace std;


Exchange::Exchange(int num_classes,
                   string fname,
                   string vocab_fname,
                   unsigned int top_word_classes,
                   bool word_boundary)
    : m_num_classes(num_classes+2),
      m_word_boundary(word_boundary)
{
    m_num_special_classes = word_boundary ? 3 : 2;
    if (fname.length()) {
        read_corpus(fname, vocab_fname);
        initialize_classes_by_freq(top_word_classes);
        set_class_counts();
    }
}


Exchange::Exchange(string fname,
                   string vocab_fname,
                   string class_fname,
                   unsigned int top_word_classes,
                   bool word_boundary)
    : m_word_boundary(word_boundary)
{
    m_num_special_classes = word_boundary ? 3 : 2;
    if (fname.length()) {
        read_corpus(fname, vocab_fname);
        read_class_initialization(class_fname);
        set_class_counts();
    }
}


Exchange::Exchange(int num_classes,
                   const map<string, int> &word_classes,
                   string fname,
                   string vocab_fname)
    : m_num_classes(num_classes+2)
{
    m_num_special_classes = 2;
    if (fname.length()) {
        read_corpus(fname, vocab_fname);
        initialize_classes_preset(word_classes);
        set_class_counts();
    }
}


void
Exchange::read_corpus(string fname,
                      string vocab_fname)
{
    string line;

    cerr << "Reading vocabulary..";
    set<string> word_types;
    SimpleFileInput corpusf(fname);
    while (corpusf.getline(line)) {
        stringstream ss(line);
        string token;
        while (ss >> token) word_types.insert(token);
    }

    if (vocab_fname.length()) {
        set<string> constrained_vocab;
        SimpleFileInput vocabf(vocab_fname);
        while (vocabf.getline(line)) {
            stringstream ss(line);
            string token;
            while (ss >> token) constrained_vocab.insert(token);
        }

        set<string> intersection;
        set_intersection(word_types.begin(), word_types.end(),
                         constrained_vocab.begin(), constrained_vocab.end(),
                         inserter(intersection, intersection.begin()));
        word_types = intersection;
    }
    cerr << " " << word_types.size() << " words" << endl;

    m_vocabulary.push_back("<s>");
    m_vocabulary_lookup["<s>"] = m_vocabulary.size() - 1;
    m_vocabulary.push_back("</s>");
    m_vocabulary_lookup["</s>"] = m_vocabulary.size() - 1;
    m_vocabulary.push_back("<unk>");
    m_vocabulary_lookup["<unk>"] = m_vocabulary.size() - 1;
    for (auto wit=word_types.begin(); wit != word_types.end(); ++wit) {
        if (wit->find("<") != string::npos && *wit != "<w>") continue;
        m_vocabulary.push_back(*wit);
        m_vocabulary_lookup[*wit] = m_vocabulary.size() - 1;
    }
    word_types.clear();

    cerr << "Reading word counts..";
    m_word_counts.resize(m_vocabulary.size());
    m_word_bigram_counts.resize(m_vocabulary.size());
    m_word_rev_bigram_counts.resize(m_vocabulary.size());
    SimpleFileInput corpusf2(fname);
    int num_tokens = 0;

    int ss_idx = m_vocabulary_lookup["<s>"];
    int se_idx = m_vocabulary_lookup["</s>"];
    int unk_idx = m_vocabulary_lookup["<unk>"];

    while (corpusf2.getline(line)) {
        vector<int> sent;
        stringstream ss(line);
        string token;

        sent.push_back(ss_idx);
        while (ss >> token) {
            if (m_word_boundary && token == "<w>") continue;
            auto vlit = m_vocabulary_lookup.find(token);
            if (vlit != m_vocabulary_lookup.end())
                sent.push_back(vlit->second);
            else sent.push_back(unk_idx);
        }
        sent.push_back(se_idx);

        for (unsigned int i=0; i<sent.size(); i++)
            m_word_counts[sent[i]]++;
        for (unsigned int i=0; i<sent.size()-1; i++) {
            m_word_bigram_counts[sent[i]][sent[i+1]]++;
            m_word_rev_bigram_counts[sent[i+1]][sent[i]]++;
        }
        num_tokens += sent.size()-2;
    }
    cerr << " " << num_tokens << " tokens" << endl;
}


void
Exchange::write_class_mem_probs(string fname) const
{
    SimpleFileOutput mfo(fname);
    mfo << "<s>\t" << START_CLASS << " " << "0.000000" << "\n";
    mfo << "<unk>\t" << UNK_CLASS << " " << "0.000000" << "\n";
    if (m_word_boundary) mfo << "<w>\t" << WB_CLASS << " " << "0.000000" << "\n";

    for (unsigned int widx = 0; widx < m_vocabulary.size(); widx++) {
        string word = m_vocabulary[widx];
        if (word.find("<") != string::npos && word != "<w>") continue;
        if (m_word_boundary && word == "<w>") continue;
        double lp = log(m_word_counts[widx]);
        lp -= log(m_class_counts[m_word_classes[widx]]);
        mfo << word << "\t" << m_word_classes[widx] << " " << lp << "\n";
    }
    mfo.close();
}


void
Exchange::write_classes(string fname) const
{
    SimpleFileOutput mfo(fname);
    assert(m_classes.size() == static_cast<unsigned int>(m_num_classes));
    for (int cidx = 0; cidx < m_num_classes; cidx++) {
        const set<int> &words = m_classes[cidx];
        for (auto wit=words.begin(); wit != words.end(); ++wit) {
            mfo << m_vocabulary[*wit] << " " << cidx << "\n";
        }
    }
    mfo.close();
}


void
Exchange::initialize_classes_by_freq(unsigned int top_word_classes)
{
    multimap<int, int> sorted_words;
    for (unsigned int i=0; i<m_word_counts.size(); ++i) {
        if (m_vocabulary[i].find("<") != string::npos && m_vocabulary[i] != "<w>") continue;
        sorted_words.insert(make_pair(m_word_counts[i], i));
    }

    m_classes.resize(m_num_classes);
    m_word_classes.resize(m_vocabulary.size(), -1);

    if (top_word_classes > 0) {
        unsigned int widx = 0;
        for (auto swit=sorted_words.rbegin(); swit != sorted_words.rend(); ++swit) {
            m_word_classes[swit->second] = widx + m_num_special_classes;
            m_classes[widx+m_num_special_classes].insert(swit->second);
            if (++widx >= top_word_classes) break;
        }
    }

    unsigned int class_idx_helper = m_num_special_classes + top_word_classes;
    for (auto swit=sorted_words.rbegin(); swit != sorted_words.rend(); ++swit) {
        if (m_word_classes[swit->second] != -1) continue;

        unsigned int class_idx = class_idx_helper % m_num_classes;
        m_word_classes[swit->second] = class_idx;
        m_classes[class_idx].insert(swit->second);

        class_idx_helper++;
        while (class_idx_helper % m_num_classes < (m_num_special_classes + top_word_classes))
            class_idx_helper++;
    }

    m_word_classes[m_vocabulary_lookup["<s>"]] = START_CLASS;
    m_word_classes[m_vocabulary_lookup["</s>"]] = START_CLASS;
    m_word_classes[m_vocabulary_lookup["<unk>"]] = UNK_CLASS;
    m_classes[START_CLASS].insert(m_vocabulary_lookup["<s>"]);
    m_classes[START_CLASS].insert(m_vocabulary_lookup["</s>"]);
    m_classes[UNK_CLASS].insert(m_vocabulary_lookup["<unk>"]);
    if (m_word_boundary) {
        m_word_classes[m_vocabulary_lookup["<w>"]] = WB_CLASS;
        m_classes[WB_CLASS].insert(m_vocabulary_lookup["<w>"]);
    }
}


void
Exchange::initialize_classes_preset(const map<string, int> &word_classes)
{
    m_classes.resize(m_num_classes);
    m_word_classes.resize(m_vocabulary.size(), -1);

    for (auto wit=m_vocabulary_lookup.begin(); wit != m_vocabulary_lookup.end(); ++wit)
    {
        if (wit->first == "<s>" || wit->first == "</s>") {
            m_word_classes[wit->second] = START_CLASS;
            m_classes[START_CLASS].insert(wit->second);
        }
        else if (wit->first == "<unk>") {
            m_word_classes[wit->second] = UNK_CLASS;
            m_classes[UNK_CLASS].insert(wit->second);
        }
        else {
            if (word_classes.at(wit->first) == START_CLASS ||
                word_classes.at(wit->first) == UNK_CLASS)
            {
                cerr << "Error, assigning word to a reserved class: " << wit->first << endl;
                exit(1);
            }
            m_word_classes[wit->second] = word_classes.at(wit->first);
            m_classes[word_classes.at(wit->first)].insert(wit->second);
        }
    }
}


void
Exchange::read_class_initialization(string class_fname)
{
    cerr << "Reading class initialization from " << class_fname << endl;

    m_word_classes.resize(m_vocabulary.size());
    int sos_idx = m_vocabulary_lookup["<s>"];
    int eos_idx = m_vocabulary_lookup["</s>"];
    int unk_idx = m_vocabulary_lookup["<unk>"];
    m_word_classes[sos_idx] = START_CLASS;
    m_word_classes[eos_idx] = START_CLASS;
    m_word_classes[unk_idx] = UNK_CLASS;
    if (m_word_boundary) {
        m_classes.resize(3);
        int wb_idx = m_vocabulary_lookup["<w>"];
        m_word_classes[wb_idx] = WB_CLASS;
        m_classes[WB_CLASS].insert(wb_idx);
    }
    else {
        m_classes.resize(2);
    }
    m_classes[START_CLASS].insert(sos_idx);
    m_classes[START_CLASS].insert(eos_idx);
    m_classes[UNK_CLASS].insert(unk_idx);

    SimpleFileInput classf(class_fname);
    string line;
    map<int, int> file_to_class_idx;
    while (classf.getline(line)) {
        if (!line.length()) continue;
        stringstream ss(line);

        string word;
        ss >> word;
        if (word == "<s>" || word == "</s>" || word == "<unk>" ||
            (m_word_boundary && word == "<w>")) {
            cerr << "Warning: You have specified special tokens in the class "
                 << "initialization file. These will be ignored." << endl;
            continue;
        }
        auto vlit = m_vocabulary_lookup.find(word);
        if (vlit == m_vocabulary_lookup.end()) continue;
        int widx = vlit->second;

        int file_idx, class_idx;
        ss >> file_idx;
        auto cit = file_to_class_idx.find(file_idx);
        if (cit != file_to_class_idx.end()) {
            class_idx = cit->second;
        }
        else {
            class_idx = m_classes.size();
            m_classes.resize(class_idx+1);
            file_to_class_idx[file_idx] = class_idx;
        }

        m_classes[class_idx].insert(widx);
        m_word_classes[widx] = class_idx;
    }

    if (m_classes.size() != static_cast<unsigned int>(m_num_classes)) {
        cerr << "Warning: You have specified class count " << m_num_classes
             << ", but provided initialization for " << m_classes.size()
             << " classes. The class count will be corrected." << endl;
        m_num_classes = m_classes.size();
    }
}


void
Exchange::set_class_counts()
{
    m_class_counts.resize(m_classes.size(), 0);
    m_class_bigram_counts.resize(m_classes.size());
    for (unsigned int i=0; i<m_class_bigram_counts.size(); i++)
        m_class_bigram_counts[i].resize(m_classes.size());
    m_class_word_counts.resize(m_vocabulary.size());
    m_word_class_counts.resize(m_vocabulary.size());

    for (unsigned int i=0; i<m_word_counts.size(); i++)
        m_class_counts[m_word_classes[i]] += m_word_counts[i];
    for (unsigned int i=0; i<m_word_bigram_counts.size(); i++) {
        int src_class = m_word_classes[i];
        map<int, int> &curr_bigram_ctxt = m_word_bigram_counts[i];
        for (auto bgit = curr_bigram_ctxt.begin(); bgit != curr_bigram_ctxt.end(); ++bgit) {
            int tgt_class = m_word_classes[bgit->first];
            m_class_bigram_counts[src_class][tgt_class] += bgit->second;
            m_class_word_counts[bgit->first][src_class] += bgit->second;
            m_word_class_counts[i][tgt_class] += bgit->second;
        }
    }
}


double
Exchange::log_likelihood() const
{
    double ll = 0.0;
    for (auto cbg1=m_class_bigram_counts.cbegin(); cbg1 != m_class_bigram_counts.cend(); ++cbg1)
        for (auto cbg2=cbg1->cbegin(); cbg2 != cbg1->cend(); ++cbg2)
            if (*cbg2 != 0) ll += *cbg2 * log(*cbg2);
    for (auto wit=m_word_counts.begin(); wit != m_word_counts.end(); ++wit)
        if (*wit != 0) ll += (*wit) * log(*wit);
    for (auto cit=m_class_counts.begin(); cit != m_class_counts.end(); ++cit)
        if (*cit != 0) ll -= 2* (*cit) * log(*cit);

    return ll;
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
Exchange::evaluate_exchange(int word,
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
Exchange::do_exchange(int word,
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


double
Exchange::evaluate_merge(int class1,
                         int class2) const
{
    double cbg_ll_diff = 0.0;
    for (int i=0; i<(int)m_class_bigram_counts.size(); i++) {
        if (i==class1 || i==class2) continue;
        int count1 = m_class_bigram_counts[i][class1];
        if (count1 > 0) cbg_ll_diff -= count1 * log(count1);
        int count2 = m_class_bigram_counts[i][class2];
        if (count2 > 0) cbg_ll_diff -= count2 * log(count2);
        int count = count1 + count2;
        if (count > 0) cbg_ll_diff += count * log(count);
    }

    for (int j=0; j<(int)m_class_bigram_counts[class1].size(); j++) {
        if (j==class1 || j==class2) continue;
        int count1 = m_class_bigram_counts[class1][j];
        if (count1 > 0) cbg_ll_diff -= count1 * log(count1);
        int count2 = m_class_bigram_counts[class2][j];
        if (count2 > 0) cbg_ll_diff -= count2 * log(count2);
        int count = count1 + count2;
        if (count > 0) cbg_ll_diff += count * log(count);
    }

    int count12 = m_class_bigram_counts[class1][class2];
    if (count12 > 0) cbg_ll_diff -= count12 * log(count12);
    int count21 = m_class_bigram_counts[class2][class1];
    if (count21 > 0) cbg_ll_diff -= count21 * log(count21);
    int count11 = m_class_bigram_counts[class1][class1];
    if (count11 > 0) cbg_ll_diff -= count11 * log(count11);
    int count22 = m_class_bigram_counts[class2][class2];
    if (count22 > 0) cbg_ll_diff -= count22 * log(count22);
    int count = count11 + count12 + count21 + count22;
    if (count > 0) cbg_ll_diff += count * log(count);

    int hypo_class_count = m_class_counts[class1] + m_class_counts[class2];
    double cc_ll_diff = -2 * (hypo_class_count) * log(hypo_class_count);
    cc_ll_diff += 2 * (m_class_counts[class1]) * log(m_class_counts[class1]);
    cc_ll_diff += 2 * (m_class_counts[class2]) * log(m_class_counts[class2]);

    return (cbg_ll_diff + cc_ll_diff);
}


void
Exchange::do_merge(int class1,
                   int class2)
{
    for (auto wit=m_classes.at(class2).begin(); wit != m_classes.at(class2).end(); ++wit)
        m_classes.at(class1).insert(*wit);
    m_classes.at(class2).clear();

    for (int i=0; i<(int)m_word_classes.size(); i++)
        if (m_word_classes[i] == class2) m_word_classes[i] = class1;

    m_class_counts[class1] += m_class_counts[class2];
    m_class_counts[class2] = 0;

    for (int i=0; i<(int)m_class_bigram_counts.size(); i++) {
        m_class_bigram_counts[i][class1] += m_class_bigram_counts[i][class2];
        m_class_bigram_counts[i][class2] = 0;
    }

    for (int j=0; j<(int)m_class_bigram_counts[class2].size(); j++) {
        m_class_bigram_counts[class1][j] += m_class_bigram_counts[class2][j];
        m_class_bigram_counts[class2][j] = 0;
    }

    for (int i=0; i<(int)m_class_word_counts.size(); i++) {
        auto cit = m_class_word_counts[i].find(class2);
        if (cit != m_class_word_counts[i].end()) {
            m_class_word_counts[i][class1] += cit->second;
            m_class_word_counts[i].erase(cit);
        }
    }

    for (int i=0; i<(int)m_word_class_counts.size(); i++) {
        auto cit = m_word_class_counts[i].find(class2);
        if (cit != m_word_class_counts[i].end()) {
            m_word_class_counts[i][class1] += cit->second;
            m_word_class_counts[i].erase(cit);
        }
    }

    m_num_classes--;
}
