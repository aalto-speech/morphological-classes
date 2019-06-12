#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <functional>

#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Categories.hh"
#include "CatPerplexity.hh"
#include "Ngram.hh"

using namespace std;

bool
process_sent(
        const LNNgram& lm,
        const Categories& wcs,
        string line,
        vector<string>& sent)
{
    sent.clear();
    stringstream ss(line);
    string word;
    while (ss >> word) {
        if (word==SENTENCE_BEGIN_SYMBOL || word==SENTENCE_END_SYMBOL) continue;
        bool unk =
                word==UNK_SYMBOL || word==CAP_UNK_SYMBOL || lm.vocabulary_lookup.find(word)==lm.vocabulary_lookup.end();
        if (!unk) {
            auto cgenit = wcs.m_category_gen_probs.find(word);
            auto cmemit = wcs.m_category_mem_probs.find(word);
            if (cgenit==wcs.m_category_gen_probs.end() || cgenit->second.size()==0)
                unk = true;
            else if (cmemit==wcs.m_category_mem_probs.end() || cmemit->second.size()==0)
                unk = true;
        }
        sent.push_back(unk ? UNK_SYMBOL : word);
    }
    if (sent.size()==0) return false;
    sent.push_back(SENTENCE_END_SYMBOL);
    return true;
}

int main(int argc, char* argv[])
{

    conf::Config config;
    config("usage: catintppl [OPTION...] WORD_ARPA CAT_ARPA CGENPROBS CMEMPROBS INPUT\n")
            ('i', "weight=FLOAT", "arg", "0.5", "Interpolation weight [0.0,1.0] for the word ARPA model")
            ('r', "unk-root-node", "", "",
                    "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
            ('n', "num-tokens=INT", "arg", "100",
                    "Upper limit for the number of tokens in each position (DEFAULT: 100)")
            ('b', "prob-beam=FLOAT", "arg", "20.0", "Probability beam (DEFAULT: 20.0)")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=5)
        config.print_help(stderr, 1);

    string arpafname = config.arguments[0];
    string cngramfname = config.arguments[1];
    string cgenpfname = config.arguments[2];
    string cmempfname = config.arguments[3];
    string infname = config.arguments[4];

    bool root_unk_states = config["unk-root-node"].specified;
    int max_tokens = config["num-tokens"].get_int();
    double prob_beam = config["prob-beam"].get_float();

    double iw = config["weight"].get_float();
    if (iw<0.0 || iw>1.0) {
        cerr << "Invalid interpolation weight: " << iw << endl;
        exit(EXIT_FAILURE);
    }
    cerr << "Interpolation weight: " << iw << endl;
    double word_iw = log(iw);
    double cat_iw = log(1.0-iw);

    LNNgram lm;
    lm.read_arpa(arpafname);

    Categories wcs;
    cerr << "Reading category generation probs.." << endl;
    wcs.read_category_gen_probs(cgenpfname);
    cerr << "Reading category membership probs.." << endl;
    wcs.read_category_mem_probs(cmempfname);

    cerr << "Reading category n-gram model.." << endl;
    LNNgram cngram;
    cngram.read_arpa(cngramfname);
    vector<int> indexmap = get_class_index_map(wcs.num_categories(), cngram);

    unsigned long int num_vocab_words = 0;
    unsigned long int num_oov_words = 0;
    unsigned long int num_sents = 0;

    SimpleFileInput corpusf(infname);
    double total_ll = 0.0;
    string line;
    while (corpusf.getline(line)) {
        vector<string> sent;
        if (!process_sent(lm, wcs, line, sent)) continue;
        CatPerplexity::CategoryHistory history(cngram);
        int curr_lm_node = lm.sentence_start_node;
        for (int i = 0; i<(int) sent.size(); i++) {
            if (sent[i]==UNK_SYMBOL) {
                if (root_unk_states)
                    curr_lm_node = lm.root_node;
                else
                    curr_lm_node = lm.advance(curr_lm_node, lm.unk_symbol_idx);
                CatPerplexity::likelihood(cngram, wcs, indexmap,
                        num_vocab_words, num_oov_words,
                        sent[i], history,
                        false, max_tokens, prob_beam);
            }
            else {
                double ngram_lp = word_iw;
                curr_lm_node = lm.score(curr_lm_node, lm.vocabulary_lookup.at(sent[i]), ngram_lp);

                double cat_lp = cat_iw+CatPerplexity::likelihood(cngram, wcs, indexmap,
                        num_vocab_words, num_oov_words,
                        sent[i], history,
                        false, max_tokens, prob_beam);

                total_ll += add_log_domain_probs(ngram_lp, cat_lp);
            }
        }
        num_sents++;
    }

    cerr << "Number of sentences processed: " << num_sents << endl;
    cerr << "Number of in-vocabulary word tokens without sentence ends: " << num_vocab_words << endl;
    cerr << "Number of in-vocabulary word tokens with sentence ends: " << num_vocab_words+num_sents << endl;
    cerr << "Number of out-of-vocabulary word tokens: " << num_oov_words << endl;
    cerr << "Total log likelihood (ln): " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/2.302585092994046 << endl;

    double ppl = exp(-1.0/double(num_vocab_words+num_sents)*total_ll);
    cerr << "Perplexity: " << ppl << endl;

    exit(EXIT_SUCCESS);
}
