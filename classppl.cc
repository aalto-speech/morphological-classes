#include <fstream>
#include <iostream>
#include <sstream>

#include "str.hh"
#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Classes.hh"
#include "Ngram.hh"

using namespace std;


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: classppl [OPTION...] CLASS_ARPA WORD_PROBS INPUT\n")
    ('r', "use-root-node", "", "", "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 3) config.print_help(stderr, 1);

    string ngramfname = config.arguments[0];
    string wordpfname = config.arguments[1];
    string infname = config.arguments[2];

    bool root_unk_states = config["use-root-node"].specified;

    WordClasses wcs;
    cerr << "Reading word probs.." << endl;
    wcs.read_word_probs(wordpfname);

    cerr << "Asserting class membership probabilities.." << endl;
    if (!wcs.assert_word_probs()) {
        cerr << "Problem in class membership probabilities" << endl;
        //exit(1);
    }

    cerr << "Reading class n-gram model.." << endl;
    Ngram ng;
    ng.read_arpa(ngramfname);
    int order = ng.order();
    int class_lm_start_node = ng.advance(ng.root_node, ng.vocabulary_lookup.at("<s>"));

    string unk;
    int unk_id;
    if (ng.vocabulary_lookup.find("<unk>") != ng.vocabulary_lookup.end()) {
        unk.assign("<unk>");
        unk_id = ng.vocabulary_lookup["<unk>"];
    }
    else if (ng.vocabulary_lookup.find("<UNK>") != ng.vocabulary_lookup.end()) {
        unk.assign("<UNK>");
        unk_id = ng.vocabulary_lookup["<UNK>"];
    }
    else {
        cerr << "Unk symbol not found in language model." << endl;
        exit(1);
    }
    cerr << "Found unknown symbol: " << unk << endl;

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        if (ng.vocabulary_lookup.find(int2str(i)) != ng.vocabulary_lookup.end())
            indexmap[i] = ng.vocabulary_lookup[int2str(i)];

    cerr << "Scoring sentences.." << endl;
    SimpleFileInput infile(infname);
    string line;
    long int num_words = 0;
    long int num_sents = 0;
    long int num_oovs = 0;
    double total_ll = 0.0;
    int linei = 0;
    while (infile.getline(line)) {

        line = str::cleaned(line);
        if (line.length() == 0) continue;
        if (++linei % 10000 == 0) cerr << "sentence " << linei << endl;

        stringstream ss(line);
        vector<string> words;
        string word;
        while (ss >> word) {
            if (word == "<s>") continue;
            if (word == "</s>") continue;
            if (wcs.m_class_memberships.find(word) == wcs.m_class_memberships.end()
                || word == "<unk>"  || word == "<UNK>")
            {
                words.push_back(unk);
                num_oovs++;
            }
            else {
                words.push_back(word);
                num_words++;
            }
        }
        num_words++;

        double sent_ll = 0.0;

        int curr_node = class_lm_start_node;
        for (int i=0; i<words.size(); i++) {
            if (words[i] == unk) {
                if (root_unk_states) curr_node = ng.root_node;
                else curr_node = ng.advance(curr_node, indexmap[UNK_CLASS]);
                continue;
            }

            const WordClassProbs &wcp = wcs.m_class_memberships.at(words[i]);
            assert(wcp.size() == 1);
            sent_ll += wcp.begin()->second;
            double ngram_score = 0.0;
            curr_node = ng.score(curr_node, indexmap[wcp.begin()->first], ngram_score);
            sent_ll += log(10.0) * ngram_score;
        }

        double ngram_score = 0.0;
        curr_node = ng.score(curr_node, ng.vocabulary_lookup.at("</s>"), ngram_score);
        sent_ll += log(10.0) * ngram_score;

        total_ll += sent_ll;
        num_sents++;
    }

    double ppl = exp(-1.0/double(num_words) * total_ll);
    cerr << endl;
    cerr << "Number of sentences: " << num_sents << endl;
    cerr << "Number of in-vocabulary words excluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oovs << endl;
    cerr << "Total log likelihood: " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/log(10.0) << endl;
    cerr << "Perplexity: " << ppl << endl;

    exit(0);
}

