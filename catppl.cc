#include <fstream>
#include <iostream>
#include <sstream>

#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Categories.hh"
#include "Ngram.hh"
#include "CatPerplexity.hh"

using namespace std;


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catppl [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT\n")
    ('u', "use-ngram-unk-states", "", "", "Use unk symbols in category n-gram contexts with unks, DEFAULT: use root node")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 4) config.print_help(stderr, 1);

    string ngramfname = config.arguments[0];
    string cgenpfname = config.arguments[1];
    string cmempfname = config.arguments[2];
    string infname = config.arguments[3];

    bool ngram_unk_states = config["use-ngram-unk-states"].specified;

    Categories wcs;
    cerr << "Reading category probs.." << endl;
    wcs.read_category_gen_probs(cgenpfname);
    cerr << "Reading word probs.." << endl;
    wcs.read_category_mem_probs(cmempfname);

    cerr << "Asserting category generation probabilities.." << endl;
    if (!wcs.assert_category_gen_probs()) {
        cerr << "Problem in category generation probabilities" << endl;
        exit(1);
    }

    cerr << "Asserting category membership probabilities.." << endl;
    if (!wcs.assert_category_mem_probs()) {
        cerr << "Problem in category membership probabilities" << endl;
        exit(1);
    }

    cerr << "Reading category n-gram model.." << endl;
    Ngram ng;
    ng.read_arpa(ngramfname);

    // The category indexes are stored as strings in the n-gram category
    vector<int> indexmap(wcs.num_categories());
    for (int i=0; i<(int)indexmap.size(); i++)
        indexmap[i] = ng.vocabulary_lookup[int2str(i)];

    cerr << "Reading input.." << endl;
    SimpleFileInput infile(infname);
    string line;
    long int num_words = 0;
    long int num_sents = 0;
    long int num_oov = 0;
    double total_ll = 0.0;
    int linei = 0;
    while (infile.getline(line)) {

        linei++;
        if (linei % 10000 == 0) cerr << "sentence " << linei << endl;

        int sent_words = 0;
        int sent_oovs = 0;
        flt_type sent_ll = likelihood(line, ng, wcs, indexmap, sent_words, sent_oovs, ngram_unk_states);
        total_ll += sent_ll;
        num_sents++;
        num_words += sent_words;
        num_oov += sent_oovs;
    }

    double ppl = exp(-1.0/double(num_words) * total_ll);
    cerr << endl;
    cerr << "Number of sentences: " << num_sents << endl;
    cerr << "Number of in-vocabulary words excluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oov << endl;
    cerr << "Total log likelihood: " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/2.302585092994046 << endl;
    cerr << "Perplexity: " << ppl << endl;

    exit(0);
}

