#include <sstream>
#include <cmath>

#include "Merging.hh"
#include "io.hh"
#include "defs.hh"

using namespace std;


Merging::Merging()
    : m_num_classes(0),
      m_num_special_classes(2)
{
}


Merging::Merging(int num_classes)
    : m_num_classes(num_classes+2),
      m_num_special_classes(2)
{
}


Merging::Merging(int num_classes,
                 const map<string, int> &word_classes,
                 string corpus_fname)
    : m_num_classes(num_classes+2),
      m_num_special_classes(2)
{
    initialize_classes_preset(word_classes);
    read_corpus(corpus_fname);
}


void
Merging::read_corpus(string fname)
{
    cerr << "Reading corpus.." << endl;
    m_word_counts.resize(m_vocabulary.size());
    m_word_bigram_counts.resize(m_vocabulary.size());
    m_word_rev_bigram_counts.resize(m_vocabulary.size());
    SimpleFileInput corpusf2(fname);

    int ss_idx = m_vocabulary_lookup["<s>"];
    int se_idx = m_vocabulary_lookup["</s>"];
    int unk_idx = m_vocabulary_lookup["<unk>"];

    string line;
    set<string> unk_types;
    unsigned long int num_tokens = 0;
    unsigned long int num_iv_tokens = 0;
    unsigned long int num_unk_tokens = 0;
    while (corpusf2.getline(line)) {
        vector<int> sent;
        stringstream ss(line);
        string token;

        sent.push_back(ss_idx);
        while (ss >> token) {
            auto vlit = m_vocabulary_lookup.find(token);
            if (vlit != m_vocabulary_lookup.end()) {
                sent.push_back(vlit->second);
                num_iv_tokens++;
            }
            else {
                sent.push_back(unk_idx);
                unk_types.insert(token);
                num_unk_tokens++;
            }
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

    cerr << "number of word tokens: " << num_tokens << endl;
    cerr << "number of in-vocabulary tokens: " << num_iv_tokens << endl;
    cerr << "number of out-of-vocabulary tokens: " << num_unk_tokens << endl;

    cerr << "Setting class counts.." << endl;
    set_class_counts();
}


void
Merging::write_class_mem_probs(string fname) const
{
    SimpleFileOutput mfo(fname);
    mfo << "<s>\t" << START_CLASS << " " << "0.000000" << "\n";
    mfo << "<unk>\t" << UNK_CLASS << " " << "0.000000" << "\n";

    for (unsigned int widx = 0; widx < m_vocabulary.size(); widx++) {
        string word = m_vocabulary[widx];
        if (word == "<s>" || word == "</s>" || word == "<unk>") continue;
        double lp = log(m_word_counts[widx]);
        lp -= log(m_class_counts[m_word_classes[widx]]);
        mfo << word << "\t" << m_word_classes[widx] << " " << lp << "\n";
    }
    mfo.close();
}


void
Merging::initialize_classes_preset(const map<string, int> &word_classes)
{
    int sos_idx = insert_word_to_vocab("<s>");
    int eos_idx = insert_word_to_vocab("</s>");
    int unk_idx = insert_word_to_vocab("<unk>");
    m_word_classes[sos_idx] = START_CLASS;
    m_word_classes[eos_idx] = START_CLASS;
    m_word_classes[unk_idx] = UNK_CLASS;
    m_classes.resize(2);
    m_classes[START_CLASS].insert(sos_idx);
    m_classes[START_CLASS].insert(eos_idx);
    m_classes[UNK_CLASS].insert(unk_idx);

    for (auto wit=word_classes.begin(); wit != word_classes.end(); ++wit) {
        string word = wit->first;
        if (word == "<s>" || word == "</s>" || word == "<unk>") {
            cerr << "Warning: You have specified special tokens in the class "
                 << "initialization. These will be ignored." << endl;
            continue;
        }

        int word_idx = insert_word_to_vocab(word);
        int class_idx = wit->second;
        if (class_idx+1 > (int)m_classes.size())
            m_classes.resize(class_idx+1);
        m_classes[class_idx].insert(word_idx);
        m_word_classes[word_idx] = class_idx;
    }

    m_num_classes = m_classes.size();
    cerr << "Read initialization for " << word_classes.size() << " words" << endl;
    cerr << m_num_classes << " classes specified" << endl;
}


int
Merging::insert_word_to_vocab(string word)
{
    auto vlit = m_vocabulary_lookup.find(word);
    if (vlit != m_vocabulary_lookup.end())
        return vlit->second;
    else {
        int word_idx = m_vocabulary.size();
        m_vocabulary.push_back(word);
        m_word_classes.push_back(-1);
        m_vocabulary_lookup[word] = word_idx;
        return word_idx;
    }
}


map<int, int>
Merging::read_class_initialization(string class_fname)
{
    cerr << "Reading class initialization from " << class_fname << endl;

    int sos_idx = insert_word_to_vocab("<s>");
    int eos_idx = insert_word_to_vocab("</s>");
    int unk_idx = insert_word_to_vocab("<unk>");
    m_word_classes[sos_idx] = START_CLASS;
    m_word_classes[eos_idx] = START_CLASS;
    m_word_classes[unk_idx] = UNK_CLASS;
    m_classes.resize(2);
    m_classes[START_CLASS].insert(sos_idx);
    m_classes[START_CLASS].insert(eos_idx);
    m_classes[UNK_CLASS].insert(unk_idx);

    SimpleFileInput classf(class_fname);
    string line;
    map<int, int> file_to_class_idx;
    int wordc = 0;
    int num_ignored_lines = 0;
    while (classf.getline(line)) {
        if (!line.length()) continue;
        stringstream ss(line);

        string word;
        int file_idx;
        ss >> word >> file_idx;
        if (ss.fail()) {
            num_ignored_lines++;
            continue;
        }
        if (word == "<s>" || word == "</s>" || word == "<unk>") {
            cerr << "Warning: You have specified special tokens in the class "
                 << "initialization file. These will be ignored." << endl;
            continue;
        }

        int widx = insert_word_to_vocab(word);
        int class_idx;
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
        wordc++;
    }

    m_num_classes = m_classes.size();
    cerr << "Read initialization for " << wordc << " words" << endl;
    cerr << m_num_classes << " classes specified" << endl;
    if (num_ignored_lines > 0)
        cerr << num_ignored_lines << " lines were ignored." << endl;

    return file_to_class_idx;
}


void
Merging::set_class_counts()
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
Merging::log_likelihood() const
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


double
Merging::evaluate_merge(int class1,
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
Merging::do_merge(int class1,
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
