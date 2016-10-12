#include <fstream>
#include <iostream>
#include <sstream>

#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Classes.hh"
#include "Ngram.hh"
#include "ClassPerplexity.hh"

using namespace std;


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: cat2ppl2 [OPTION...] CLASS_ARPA CLASS_PROBS WORD_PROBS INPUT\n")
    ('u', "use-ngram-unk-states", "", "", "Use unk symbols in class n-gram contexts with unks, DEFAULT: use root node")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 4) config.print_help(stderr, 1);

    string ngramfname = config.arguments[0];
    string classpfname = config.arguments[1];
    string wordpfname = config.arguments[2];
    string infname = config.arguments[3];

    bool ngram_unk_states = config["use-ngram-unk-states"].specified;

    WordClasses wcs;
    cerr << "Reading class probs.." << endl;
    wcs.read_class_probs(classpfname);
    cerr << "Reading word probs.." << endl;
    wcs.read_word_probs(wordpfname);

    cerr << "Asserting class generation probabilities.." << endl;
    if (!wcs.assert_class_probs()) {
        cerr << "Problem in class generation probabilities" << endl;
        exit(1);
    }

    cerr << "Asserting class membership probabilities.." << endl;
    if (!wcs.assert_word_probs()) {
        cerr << "Problem in class membership probabilities" << endl;
        //exit(1);
    }

    cerr << "Reading class n-gram model.." << endl;
    Ngram ng;
    ng.read_arpa(ngramfname);
    int order = ng.order();

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_classes());
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

