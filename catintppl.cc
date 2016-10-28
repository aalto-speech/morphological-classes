#include <fstream>
#include <iostream>
#include <sstream>

#include "defs.hh"
#include "conf.hh"
#include "Ngram.hh"
#include "CatPerplexity.hh"

using namespace std;


flt_type word_likelihoods(Ngram &lm,
                          int unk_id,
                          string line,
                          int &num_words,
                          int &num_oov,
                          vector<flt_type> &word_lls)
{
    word_lls.clear();

    string start_symbol = "<s>";
    string end_symbol = "</s>";

    stringstream ss(line);
    vector<string> words;
    string word;
    while (ss >> word) {
        if (word == start_symbol) continue;
        if (word == end_symbol) continue;
        words.push_back(word);
    }
    words.push_back(end_symbol);

    double sent_ll = 0.0;
    int node_id = lm.advance(lm.root_node, lm.vocabulary_lookup[start_symbol]);
    for (auto wit=words.begin(); wit != words.end(); ++wit) {
        double score = 0.0;
        if (lm.vocabulary_lookup.find(*wit) != lm.vocabulary_lookup.end()
            && lm.vocabulary_lookup.at(*wit) != unk_id)
        {
            int sym = lm.vocabulary_lookup[*wit];
            node_id = lm.score(node_id, sym, score);
            score *= log(10.0);
            sent_ll += score;
            word_lls.push_back(score);
            num_words++;
        }
        else {
            node_id = lm.score(node_id, unk_id, score); // VariKN style UNKs
            /* node_id = lm.root_node; SRILM style UNKs */
            num_oov++;
        }
    }

    return sent_ll;
}



int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catintppl [OPTION...] ARPAFILE CLASS_ARPA CLASS_PROBS WORD_PROBS INPUT\n")
    ('w', "weight=FLOAT", "arg", "0.5", "Interpolation weight [0.0,1,0] for the word ARPA model")
    ('u', "use-ngram-unk-states", "", "", "Use unk symbols in class n-gram contexts with unks, DEFAULT: use root node")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 5) config.print_help(stderr, 1);

    string arpafname = config.arguments[0];
    string classngramfname = config.arguments[1];
    string classpfname = config.arguments[2];
    string wordpfname = config.arguments[3];
    string infname = config.arguments[4];

    bool ngram_unk_states = config["use-ngram-unk-states"].specified;

    double iw = config["weight"].get_float();
    if (iw < 0.0 || iw > 1.0) {
        cerr << "Invalid interpolation weight: " << iw << endl;
        exit(1);
    }
    cerr << "Interpolation weight: " << iw << endl;
    double word_iw = log(iw);
    double class_iw = log(1.0-iw);

    Ngram lm;
    lm.read_arpa(arpafname);

    string unk;
    int unk_id;
    if (lm.vocabulary_lookup.find("<unk>") != lm.vocabulary_lookup.end()) {
        unk.assign("<unk>");
        unk_id = lm.vocabulary_lookup["<unk>"];
    }
    else if (lm.vocabulary_lookup.find("<UNK>") != lm.vocabulary_lookup.end()) {
        unk.assign("<UNK>");
        unk_id = lm.vocabulary_lookup["<UNK>"];
    }
    else {
        cerr << "Unk symbol not found in language model." << endl;
        exit(1);
    }

    Categories wcs;
    cerr << "Reading class probs.." << endl;
    wcs.read_class_gen_probs(classpfname);
    cerr << "Reading word probs.." << endl;
    wcs.read_class_mem_probs(wordpfname);

    cerr << "Asserting class generation probabilities.." << endl;
    if (!wcs.assert_class_gen_probs()) {
        cerr << "Problem in class generation probabilities" << endl;
        exit(1);
    }

    cerr << "Asserting class membership probabilities.." << endl;
    if (!wcs.assert_class_mem_probs()) {
        cerr << "Problem in class membership probabilities" << endl;
        //exit(1);
    }

    cerr << "Reading class n-gram model.." << endl;
    Ngram class_ng;
    class_ng.read_arpa(classngramfname);

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        indexmap[i] = class_ng.vocabulary_lookup[int2str(i)];

    SimpleFileInput infile(infname);
    string line;
    long int num_words = 0;
    long int num_oov = 0;
    long int num_sents = 0;
    long int num_skipped_sents = 0;
    double total_ll = 0.0;
    int linei = 0;
    while (infile.getline(line)) {

        linei++;
        if (linei % 10000 == 0) cerr << "sentence " << linei << endl;

        int numww=0, numow=0;
        vector<flt_type> ngram_lls;
        word_likelihoods(lm, unk_id, line,
                         numww, numow, ngram_lls);

        int numwc=0, numoc=0;
        vector<flt_type> class_lls;
        likelihood(line, class_ng, wcs, indexmap,
                   class_lls, numwc, numoc, ngram_unk_states);

        if (ngram_lls.size() != class_lls.size()
            || numww != numwc || numow != numoc)
        {
            /*
            cerr << "ngram lls: " << ngram_lls.size() << "\tngram words/oovs: " << numww << "/" << numow << endl ;
            cerr << "class lls: " << class_lls.size() << "\tclass words/oovs: " << numwc << "/" << numoc << endl ;
            cerr << "Problem in interpolating likelihoods" << endl;
            cerr << line << endl;
            */

            num_skipped_sents++;
            continue;
        }

        flt_type sent_ll = 0.0;
        for (int i=0; i<(int)ngram_lls.size(); i++) {
            flt_type word_ll = ngram_lls[i] + word_iw;
            flt_type class_ll = class_lls[i] + class_iw;
            flt_type ll = add_log_domain_probs(word_ll, class_ll);
            sent_ll += ll;
        }

        num_words += numww;
        num_oov += numow;

        total_ll += sent_ll;
        num_sents++;
    }

    double ppl = exp(-1.0/double(num_words) * total_ll);
    cerr << endl;
    cerr << "Number of sentences: " << num_sents << endl;
    cerr << "Number of skipped sentences: " << num_skipped_sents << endl;
    cerr << "Number of in-vocabulary words excluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oov << endl;
    cerr << "Total log likelihood (ln): " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/2.302585092994046 << endl;
    cerr << "Perplexity: " << ppl << endl;

    exit(0);
}

