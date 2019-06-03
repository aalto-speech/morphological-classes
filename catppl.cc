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

double
catppl(string corpusfname,
        const LNNgram& cngram,
        const vector<int>& indexmap,
        const Categories& categories,
        unsigned long int& num_vocab_words,
        unsigned long int& num_oov_words,
        unsigned long int& num_sents,
        bool root_unk_states = false,
        int max_tokens = 100,
        double beam = 20.0)
{
    SimpleFileInput corpusf(corpusfname);
    double total_ll = 0.0;
    string line;
    while (corpusf.getline(line)) {
        vector<string> sent;
        if (!CatPerplexity::process_sent(line, sent)) continue;
        CatPerplexity::CategoryHistory history(cngram);
        for (auto wit = sent.begin(); wit!=sent.end(); ++wit)
            total_ll +=
                    CatPerplexity::likelihood(cngram, categories, indexmap,
                            num_vocab_words, num_oov_words,
                            *wit, history,
                            false, max_tokens, beam);
        num_sents++;
    }

    return total_ll;
}

int main(int argc, char* argv[])
{

    conf::Config config;
    config("usage: catppl [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT\n")
            ('r', "unk-root-node", "", "",
                    "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
            ('n', "num-tokens=INT", "arg", "100",
                    "Upper limit for the number of tokens in each position (DEFAULT: 100)")
            ('b', "prob-beam=FLOAT", "arg", "20.0", "Probability beam (DEFAULT: 20.0)")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=4)
        config.print_help(stderr, 1);

    string cngramfname = config.arguments[0];
    string cgenpfname = config.arguments[1];
    string cmempfname = config.arguments[2];
    string infname = config.arguments[3];

    bool root_unk_states = config["unk-root-node"].specified;
    int max_tokens = config["num-tokens"].get_int();
    double prob_beam = config["prob-beam"].get_float();

    Categories wcs;
    cerr << "Reading category generation probs.." << endl;
    wcs.read_category_gen_probs(cgenpfname);
    cerr << "Reading category membership probs.." << endl;
    wcs.read_category_mem_probs(cmempfname);

    cerr << "Reading category n-gram model.." << endl;
    LNNgram cngram;
    cngram.read_arpa(cngramfname);

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_categories());
    for (int i = 0; i<(int) indexmap.size(); i++)
        if (cngram.vocabulary_lookup.find(int2str(i))!=cngram.vocabulary_lookup.end())
            indexmap[i] = cngram.vocabulary_lookup[int2str(i)];
        else
            cerr << "warning, category not found in the n-gram: " << i << endl;

    unsigned long int num_vocab_words = 0;
    unsigned long int num_oov_words = 0;
    unsigned long int num_sents = 0;
    double total_ll = catppl(infname,
            cngram, indexmap, wcs,
            num_vocab_words, num_oov_words, num_sents,
            root_unk_states, max_tokens, prob_beam);

    cout << "Number of sentences processed: " << num_sents << endl;
    cout << "Number of in-vocabulary word tokens without sentence ends: " << num_vocab_words << endl;
    cout << "Number of in-vocabulary word tokens with sentence ends: " << num_vocab_words+num_sents << endl;
    cout << "Number of out-of-vocabulary word tokens: " << num_oov_words << endl;
    cout << "Likelihood: " << total_ll << endl;
    double ppl = exp(-1.0/double(num_vocab_words+num_sents)*total_ll);
    cout << "Perplexity: " << ppl << endl;

    exit(EXIT_SUCCESS);
}
