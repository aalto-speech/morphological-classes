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
process_sent(string line,
             const TrainingParameters &params,
             vector<string> &sent)
{
    sent.clear();
    stringstream ss(line);
    string word;
    while (ss >> word) {
        if (word == "<s>" || word == "</s>") continue;
        sent.push_back(word);
    }
    sent.push_back("</s>");
    if (sent.size() > params.max_line_length) return false;
    if (sent.size() == 0) return false;
    return true;
}


double
catppl(string corpusfname,
       const Ngram &cngram,
       const vector<int> &indexmap,
       const Categories &categories,
       const TrainingParameters &params,
       unsigned long int &num_vocab_words,
       unsigned long int &num_oov_words,
       unsigned long int &num_sents,
       bool verbose)
{
    SimpleFileInput corpusf(corpusfname);
    double total_ll = 0.0;
    string line;
    while (corpusf.getline(line)) {
        vector<string> sent;
        if (!process_sent(line, params, sent)) continue;
        CatPerplexity::CategoryHistory history(cngram);
        for (int i = 0; i < (int)sent.size(); i++)
            total_ll +=
                    CatPerplexity::likelihood(cngram, categories, indexmap,
                                              num_vocab_words, num_oov_words,
                                              sent[i], history,
                                              true, params.num_tokens, params.prob_beam);
        num_sents++;
        if (verbose && num_sents % 5000 == 0) cerr << num_sents << endl;
    }

    return total_ll;
}


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catppl [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT\n")
    ('n', "num-tokens=INT", "arg", "100", "Upper limit for the number of tokens in each position (DEFAULT: 100)")
    ('f', "num-final-tokens=INT", "arg", "10", "Upper limit for the number of tokens in the last position (DEFAULT: 10)")
    ('l', "max-line-length=INT", "arg", "100", "Maximum sentence length as number of words (DEFAULT: 100)")
    ('b', "prob-beam=FLOAT", "arg", "20.0", "Probability beam (default 20.0)")
    ('v', "verbose", "", "", "Print some information")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 4)
        config.print_help(stderr, 1);

    string cngramfname = config.arguments[0];
    string cgenpfname = config.arguments[1];
    string cmempfname = config.arguments[2];
    string infname = config.arguments[3];

    TrainingParameters params;
    params.num_tokens = config["num-tokens"].get_int();
    params.num_final_tokens = config["num-final-tokens"].get_int();
    params.max_line_length = config["max-line-length"].get_int();
    params.prob_beam = config["prob-beam"].get_float();
    bool verbose = config["verbose"].specified;

    Categories wcs;
    cerr << "Reading category generation probs.." << endl;
    wcs.read_category_gen_probs(cgenpfname);
    cerr << "Reading category membership probs.." << endl;
    wcs.read_category_mem_probs(cmempfname);

    cerr << "Reading category n-gram model.." << endl;
    Ngram cngram;
    cngram.read_arpa(cngramfname);
    params.max_order = cngram.max_order;

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_categories());
    for (int i = 0; i < (int) indexmap.size(); i++)
        if (cngram.vocabulary_lookup.find(int2str(i)) != cngram.vocabulary_lookup.end())
            indexmap[i] = cngram.vocabulary_lookup[int2str(i)];
        else
            cerr << "warning, category not found in the n-gram: " << i << endl;

    unsigned long int num_vocab_words = 0;
    unsigned long int num_oov_words = 0;
    unsigned long int num_sents = 0;
    double total_ll = catppl(infname,
                             cngram, indexmap, wcs,
                             params,
                             num_vocab_words, num_oov_words, num_sents,
                             verbose);

    cout << "Number of sentences processed: " << num_sents << endl;
    cout << "Number of in-vocabulary word tokens without sentence ends: " << num_vocab_words << endl;
    cout << "Number of in-vocabulary word tokens with sentence ends: " << num_vocab_words + num_sents << endl;
    cout << "Number of out-of-vocabulary word tokens: " << num_oov_words << endl;
    cout << "Likelihood: " << total_ll << endl;
    double ppl = exp(-1.0 / double(num_vocab_words + num_sents) * total_ll);
    cout << "Perplexity: " << ppl << endl;

    exit(EXIT_SUCCESS);
}
