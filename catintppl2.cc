#include <fstream>
#include <iostream>
#include <sstream>

#include "defs.hh"
#include "conf.hh"
#include "Ngram.hh"
#include "CatPerplexity.hh"

using namespace std;


void read_class_model(Categories &wcs,
                      Ngram &ng,
                      vector<int> &indexmap,
                      string arpafname,
                      string cgenfname,
                      string cmemberfname)
{
    cerr << "Reading class generation probabilities.." << endl;
    wcs.read_category_gen_probs(cgenfname);
    cerr << "Reading class membership probabilities.." << endl;
    wcs.read_category_mem_probs(cmemberfname);

    cerr << "Asserting class generation probabilities.." << endl;
    if (!wcs.assert_category_gen_probs()) {
        cerr << "Problem in class generation probabilities" << endl;
        exit(1);
    }

    cerr << "Asserting class membership probabilities.." << endl;
    if (!wcs.assert_category_mem_probs()) {
        cerr << "Problem in class membership probabilities" << endl;
        //exit(1);
    }

    cerr << "Reading class n-gram model.." << endl;
    ng.read_arpa(arpafname);

    // The class indexes are stored as strings in the n-gram class
    indexmap.resize(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        indexmap[i] = ng.vocabulary_lookup[int2str(i)];
}


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


void evaluate(Ngram &lm,
              Categories &wcs,
              Ngram &class_ng,
              vector<int> &indexmap,
              Categories &wcs2,
              Ngram &class_ng2,
              vector<int> &indexmap2,
              string infname,
              bool ngram_unk_states,
              vector<double> &weights)
{
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

    SimpleFileInput infile(infname);
    long int num_words = 0;
    long int num_oov = 0;
    long int num_sents = 0;
    long int num_skipped_sents = 0;
    double total_ll = 0.0;
    int linei = 0;
    string line;
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

        int numwc2=0, numoc2=0;
        vector<flt_type> class_lls2;
        likelihood(line, class_ng2, wcs2, indexmap2,
                   class_lls2, numwc2, numoc2, ngram_unk_states);

        if (ngram_lls.size() != class_lls.size()
            || class_lls.size() != class_lls2.size()
            || numww != numwc
            || numwc != numwc2
            || numow != numoc
            || numoc != numoc2)
        {
            num_skipped_sents++;
            continue;
        }

        flt_type sent_ll = 0.0;
        for (int i=0; i<(int)ngram_lls.size(); i++) {
            flt_type word_ll = ngram_lls[i] + weights[0];
            flt_type class_ll = class_lls[i] + weights[1];
            flt_type class_ll2 = class_lls2[i] + weights[2];
            flt_type ll = add_log_domain_probs(word_ll, class_ll);
            ll = add_log_domain_probs(ll, class_ll2);
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
    cerr << "Number of in-vocabulary words exluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oov << endl;
    cerr << "Total log likelihood: " << total_ll << endl;
    cerr << "Perplexity: " << ppl << endl;
}


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catintppl2 [OPTION...] ARPAFILE CLASS_ARPA CLASS_PROBS WORD_PROBS CLASS_ARPA2 CLASS_PROBS2 WORD_PROBS2 INPUT\n")
    ('w', "weights=FILE", "arg must", "", "File containing interpolation weights, format: three floats per line separated by whitespace")
    ('u', "use-ngram-unk-states", "", "", "Use unk symbols in class n-gram contexts with unks, DEFAULT: use root node")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 8) config.print_help(stderr, 1);

    string arpafname = config.arguments[0];
    string classngramfname = config.arguments[1];
    string classpfname = config.arguments[2];
    string wordpfname = config.arguments[3];
    string classngramfname2 = config.arguments[4];
    string classpfname2 = config.arguments[5];
    string wordpfname2 = config.arguments[6];
    string infname = config.arguments[7];

    bool ngram_unk_states = config["use-ngram-unk-states"].specified;

    vector<vector<double> > weights;
    SimpleFileInput wif(config["weights"].get_str());
    string line;
    while (wif.getline(line)) {
        if (!line.length()) continue;
        cerr << line << endl;
        stringstream ss(line);
        vector<double> w;
        double tmp;
        double total = 0.0;
        ss >> tmp; w.push_back(tmp); total += tmp;
        ss >> tmp; w.push_back(tmp); total += tmp;
        ss >> tmp; w.push_back(tmp); total += tmp;
        if (fabs(total - 1.0) > 0.000000001) {
            cerr << "problem in line: " << line << endl;
            exit(1);
        }
        for (auto wit=w.begin(); wit != w.end(); ++wit)
            *wit = log(*wit);
        weights.push_back(w);
    }

    Ngram lm;
    lm.read_arpa(arpafname);

    Categories wcs;
    Ngram class_ng;
    vector<int> indexmap;
    read_class_model(wcs, class_ng, indexmap,
                     classngramfname, classpfname, wordpfname);

    Categories wcs2;
    Ngram class_ng2;
    vector<int> indexmap2;
    read_class_model(wcs2, class_ng2, indexmap2,
                     classngramfname2, classpfname2, wordpfname2);

    for (auto wit=weights.begin(); wit != weights.end(); ++wit) {
        evaluate(lm, wcs, class_ng, indexmap,
                 wcs2, class_ng2, indexmap2,
                 infname, ngram_unk_states,
                 *wit);
    }

    exit(0);
}

