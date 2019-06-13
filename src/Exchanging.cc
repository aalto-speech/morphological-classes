#include <sstream>
#include <cmath>
#include <ctime>
#include <thread>

#include "Exchanging.hh"
#include "io.hh"
#include "defs.hh"

using namespace std;

Exchanging::Exchanging()
        :Merging()
{
}

Exchanging::Exchanging(
        int num_classes,
        const std::map<std::string, int>& word_classes,
        string corpus_fname)
        :Merging(num_classes, word_classes, corpus_fname)
{
}

Exchanging::Exchanging(
        int num_classes,
        string corpus_fname,
        string vocab_fname)
        :Merging(num_classes)
{
    initialize_classes_by_freq(corpus_fname, vocab_fname);
    read_corpus(corpus_fname);
}

void
Exchanging::initialize_classes_by_freq(
        string corpus_fname,
        string vocab_fname)
{
    cerr << "Initializing classes by frequency order from corpus " << corpus_fname << endl;

    int sos_idx = insert_word_to_vocab(SENTENCE_BEGIN_SYMBOL);
    int eos_idx = insert_word_to_vocab(SENTENCE_END_SYMBOL);
    int unk_idx = insert_word_to_vocab(UNK_SYMBOL);
    m_word_classes[sos_idx] = START_CLASS;
    m_word_classes[eos_idx] = START_CLASS;
    m_word_classes[unk_idx] = UNK_CLASS;
    m_classes.resize(m_num_classes);
    m_classes[START_CLASS].insert(sos_idx);
    m_classes[START_CLASS].insert(eos_idx);
    m_classes[UNK_CLASS].insert(unk_idx);

    set<string> constrained_vocab;
    if (vocab_fname.length()) {
        cerr << "Reading vocabulary from " << vocab_fname << endl;
        SimpleFileInput vocabf(vocab_fname);
        string line;
        while (vocabf.getline(line)) {
            stringstream ss(line);
            string token;
            while (ss >> token) constrained_vocab.insert(token);
        }
    }

    string line;
    SimpleFileInput corpusf(corpus_fname);
    map<string, int> word_counts;
    while (corpusf.getline(line)) {
        vector<int> sent;
        stringstream ss(line);
        string token;
        while (ss >> token) word_counts[token] += 1;
    }

    multimap<int, string> sorted_words;
    for (auto wit = word_counts.begin(); wit!=word_counts.end(); ++wit) {
        string word = wit->first;
        if (word==SENTENCE_BEGIN_SYMBOL || word==SENTENCE_END_SYMBOL || word==UNK_SYMBOL) continue;
        if (vocab_fname.length()>0 && constrained_vocab.find(word)==constrained_vocab.end())
            continue;
        sorted_words.insert(make_pair(wit->second, word));
    }

    unsigned int class_idx_helper = m_num_special_classes;
    for (auto swit = sorted_words.rbegin(); swit!=sorted_words.rend(); ++swit) {
        int widx = insert_word_to_vocab(swit->second);
        if (m_word_classes[widx]!=-1) continue;

        unsigned int class_idx = class_idx_helper%m_num_classes;
        m_word_classes[widx] = class_idx;
        m_classes[class_idx].insert(widx);

        class_idx_helper++;
        while (class_idx_helper%m_num_classes<(unsigned int) m_num_special_classes)
            class_idx_helper++;
    }
}

inline void
evaluate_ll_diff(double& ll_diff,
        int old_count,
        int new_count)
{
    if (old_count!=0)
        ll_diff -= old_count*log(old_count);
    if (new_count!=0)
        ll_diff += new_count*log(new_count);
}

inline int
get_count(const map<int, int>& ctxt,
        int element)
{
    auto it = ctxt.find(element);
    if (it!=ctxt.end()) return it->second;
    else return 0;
}

double
Exchanging::evaluate_exchange(
        int word,
        int curr_class,
        int tentative_class) const
{
    double ll_diff = 0.0;
    int wc = m_word_counts[word];
    const map<int, int>& wb_ctxt = m_word_bigram_counts.at(word);
    const map<int, int>& cw_counts = m_class_word_counts.at(word);
    const map<int, int>& wc_counts = m_word_class_counts.at(word);

    ll_diff += 2*(m_class_counts[curr_class])*log(m_class_counts[curr_class]);
    ll_diff -= 2*(m_class_counts[curr_class]-wc)*log(m_class_counts[curr_class]-wc);
    ll_diff += 2*(m_class_counts[tentative_class])*log(m_class_counts[tentative_class]);
    ll_diff -= 2*(m_class_counts[tentative_class]+wc)*log(m_class_counts[tentative_class]+wc);

    for (auto wcit = wc_counts.begin(); wcit!=wc_counts.end(); ++wcit) {
        if (wcit->first==curr_class) continue;
        if (wcit->first==tentative_class) continue;

        int curr_count = m_class_bigram_counts[curr_class][wcit->first];
        int new_count = curr_count-wcit->second;
        evaluate_ll_diff(ll_diff, curr_count, new_count);

        curr_count = m_class_bigram_counts[tentative_class][wcit->first];
        new_count = curr_count+wcit->second;
        evaluate_ll_diff(ll_diff, curr_count, new_count);
    }

    for (auto wcit = cw_counts.begin(); wcit!=cw_counts.end(); ++wcit) {
        if (wcit->first==curr_class) continue;
        if (wcit->first==tentative_class) continue;

        int curr_count = m_class_bigram_counts[wcit->first][curr_class];
        int new_count = curr_count-wcit->second;
        evaluate_ll_diff(ll_diff, curr_count, new_count);

        curr_count = m_class_bigram_counts[wcit->first][tentative_class];
        new_count = curr_count+wcit->second;
        evaluate_ll_diff(ll_diff, curr_count, new_count);
    }

    int self_count = 0;
    auto scit = wb_ctxt.find(word);
    if (scit!=wb_ctxt.end()) self_count = scit->second;

    int curr_count = m_class_bigram_counts[curr_class][tentative_class];
    int new_count = curr_count-get_count(wc_counts, tentative_class)
            +get_count(cw_counts, curr_class)-self_count;
    evaluate_ll_diff(ll_diff, curr_count, new_count);

    curr_count = m_class_bigram_counts[tentative_class][curr_class];
    new_count = curr_count-get_count(cw_counts, tentative_class)
            +get_count(wc_counts, curr_class)-self_count;
    evaluate_ll_diff(ll_diff, curr_count, new_count);

    curr_count = m_class_bigram_counts[curr_class][curr_class];
    new_count = curr_count-get_count(wc_counts, curr_class)
            -get_count(cw_counts, curr_class)+self_count;
    evaluate_ll_diff(ll_diff, curr_count, new_count);

    curr_count = m_class_bigram_counts[tentative_class][tentative_class];
    new_count = curr_count+get_count(wc_counts, tentative_class)
            +get_count(cw_counts, tentative_class)+self_count;
    evaluate_ll_diff(ll_diff, curr_count, new_count);

    return ll_diff;
}

void
Exchanging::do_exchange(
        int word,
        int prev_class,
        int new_class)
{
    int wc = m_word_counts[word];
    m_class_counts[prev_class] -= wc;
    m_class_counts[new_class] += wc;

    map<int, int>& bctxt = m_word_bigram_counts[word];
    for (auto wit = bctxt.begin(); wit!=bctxt.end(); ++wit) {
        if (wit->first==word) continue;
        int tgt_class = m_word_classes[wit->first];
        m_class_bigram_counts[prev_class][tgt_class] -= wit->second;
        m_class_bigram_counts[new_class][tgt_class] += wit->second;
        m_class_word_counts[wit->first][prev_class] -= wit->second;
        m_class_word_counts[wit->first][new_class] += wit->second;
    }

    map<int, int>& rbctxt = m_word_rev_bigram_counts[word];
    for (auto wit = rbctxt.begin(); wit!=rbctxt.end(); ++wit) {
        if (wit->first==word) continue;
        int src_class = m_word_classes[wit->first];
        m_class_bigram_counts[src_class][prev_class] -= wit->second;
        m_class_bigram_counts[src_class][new_class] += wit->second;
        m_word_class_counts[wit->first][prev_class] -= wit->second;
        m_word_class_counts[wit->first][new_class] += wit->second;
    }

    auto wit = bctxt.find(word);
    if (wit!=bctxt.end()) {
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
Exchanging::iterate_exchange(
        int max_iter,
        int max_seconds,
        int ll_print_interval,
        int model_write_interval,
        string model_base,
        int num_threads)
{
    time_t start_time = time(0);
    time_t last_model_write_time = start_time;
    int tmp_model_idx = 1;

    int curr_iter = 0;
    while (true) {
        cerr << "Iteration " << curr_iter+1 << endl;

        for (int widx = 0; widx<(int) m_vocabulary.size(); widx++) {

            if (m_word_classes[widx]==START_CLASS ||
                    m_word_classes[widx]==UNK_CLASS)
                continue;

            int curr_class = m_word_classes[widx];
            if (m_classes[curr_class].size()==1) continue;
            int best_class = -1;
            double best_ll_diff = -1e20;

            if (num_threads>1) {
                exchange_thr(num_threads,
                        widx,
                        curr_class,
                        best_class,
                        best_ll_diff);
            }
            else {
                for (int cidx = m_num_special_classes; cidx<(int) m_classes.size(); cidx++) {
                    if (cidx==curr_class) continue;
                    double ll_diff = evaluate_exchange(widx, curr_class, cidx);
                    if (ll_diff>best_ll_diff) {
                        best_ll_diff = ll_diff;
                        best_class = cidx;
                    }
                }
            }

            if (best_class==-1 || best_ll_diff==-1e20) {
                cerr << "problem in word: " << m_vocabulary[widx] << endl;
                exit(1);
            }

            if (best_ll_diff>0.0)
                do_exchange(widx, curr_class, best_class);

            if ((ll_print_interval>0 && widx%ll_print_interval==0)
                    || widx+1==(int) m_vocabulary.size()) {
                double ll = log_likelihood();
                cerr << "log likelihood: " << ll << endl;
            }

            if (widx%1000==0) {
                time_t curr_time = time(0);

                if (curr_time-start_time>max_seconds)
                    return log_likelihood();

                if (model_write_interval>0 && curr_time-last_model_write_time>model_write_interval) {
                    string temp_base = model_base+".temp"+int2str(tmp_model_idx);
                    write_class_mem_probs(temp_base+".cmemprobs.gz");
                    last_model_write_time = curr_time;
                    tmp_model_idx++;
                }
            }
        }

        curr_iter++;
        if (max_iter>0 && curr_iter>=max_iter) return log_likelihood();
    }
}

double
Exchanging::iterate_exchange(
        vector<vector<int>> super_classes,
        map<int, int> super_class_lookup,
        int max_iter,
        int max_seconds,
        int ll_print_interval,
        int model_write_interval,
        string model_base)
{
    time_t start_time, curr_time;
    time_t last_model_write_time;
    start_time = time(0);
    last_model_write_time = start_time;
    int tmp_model_idx = 1;

    int curr_iter = 0;
    while (true) {
        for (int widx = 0; widx<(int) m_vocabulary.size(); widx++) {

            if (m_word_classes[widx]==START_CLASS ||
                    m_word_classes[widx]==UNK_CLASS)
                continue;

            int curr_class = m_word_classes[widx];
            if (m_classes[curr_class].size()==1) continue;

            int super_class_idx = super_class_lookup[curr_class];
            vector<int>& super_class = super_classes[super_class_idx];
            if (super_class.size()<2) continue;

            int best_class = -1;
            double best_ll_diff = -1e20;
            for (auto cit = super_class.begin(); cit!=super_class.end(); ++cit) {
                if (*cit==curr_class) continue;
                double ll_diff = evaluate_exchange(widx, curr_class, *cit);
                if (ll_diff>best_ll_diff) {
                    best_ll_diff = ll_diff;
                    best_class = *cit;
                }
            }

            if (best_class==-1 || best_ll_diff==-1e20) {
                cerr << "problem in word: " << m_vocabulary[widx] << endl;
                exit(1);
            }

            if (best_ll_diff>0.0)
                do_exchange(widx, curr_class, best_class);

            if ((ll_print_interval>0 && widx%ll_print_interval==0)
                    || widx+1==(int) m_vocabulary.size()) {
                double ll = log_likelihood();
                cerr << "log likelihood: " << ll << endl;
            }

            if (widx%1000==0) {
                curr_time = time(0);

                if (curr_time-start_time>max_seconds)
                    return log_likelihood();

                if (model_write_interval>0 && curr_time-last_model_write_time>model_write_interval) {
                    string temp_base = model_base+".temp"+int2str(tmp_model_idx);
                    write_class_mem_probs(temp_base+".cmemprobs.gz");
                    last_model_write_time = curr_time;
                    tmp_model_idx++;
                }
            }
        }

        curr_iter++;
        if (max_iter>0 && curr_iter>=max_iter) return log_likelihood();
    }
}

void
Exchanging::exchange_thr_worker(
        int num_threads,
        int thread_index,
        int word_index,
        int curr_class,
        int& best_class,
        double& best_ll_diff)
{
    for (int cidx = m_num_special_classes; cidx<(int) m_classes.size(); cidx++) {
        if (cidx==curr_class) continue;
        if (cidx%num_threads!=thread_index) continue;
        double ll_diff = evaluate_exchange(word_index, curr_class, cidx);
        if (ll_diff>best_ll_diff) {
            best_ll_diff = ll_diff;
            best_class = cidx;
        }
    }
}

void
Exchanging::exchange_thr(
        int num_threads,
        int word_index,
        int curr_class,
        int& best_class,
        double& best_ll_diff)
{
    vector<double> thr_ll_diffs(num_threads, -1e20);
    vector<int> thr_best_classes(num_threads, -1);
    vector<std::thread*>workers;
    for (int t = 0; t<num_threads; t++) {
        std::thread* worker = new std::thread(&Exchanging::exchange_thr_worker, this,
                num_threads, t,
                word_index, curr_class,
                std::ref(thr_best_classes[t]),
                std::ref(thr_ll_diffs[t]));
        workers.push_back(worker);
    }
    for (int t = 0; t<num_threads; t++) {
        workers[t]->join();
        if (thr_ll_diffs[t]>best_ll_diff) {
            best_ll_diff = thr_ll_diffs[t];
            best_class = thr_best_classes[t];
        }
    }
}

