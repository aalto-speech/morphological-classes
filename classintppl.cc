#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>

#include "str.hh"
#include "defs.hh"
#include "conf.hh"
#include "Ngram.hh"
#include "ClassPerplexity.hh"

using namespace std;



void preprocess_sent(string line,
                     const WordClasses &wcs,
                     string unk_symbol,
                     vector<string> &words,
                     long int &num_words,
                     long int &num_oovs)
{
    stringstream ss(line);
    words.clear();
    string word;
    while (ss >> word) {
        if (word == "<s>") continue;
        if (word == "</s>") continue;
        if (wcs.m_class_memberships.find(word) == wcs.m_class_memberships.end()
            || word == "<unk>"  || word == "<UNK>")
        {
            words.push_back(unk_symbol);
            num_oovs++;
        }
        else {
            words.push_back(word);
            num_words++;
        }
    }
    num_words++;
}


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: classintppl [OPTION...] ARPAFILE CLASS_ARPA WORD_PROBS INPUT\n")
    ('w', "weight=FLOAT", "arg", "0.5", "Interpolation weight [0.0,1,0] for the word ARPA model")
    ('r', "use-root-node", "", "", "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 4) config.print_help(stderr, 1);

    string arpafname = config.arguments[0];
    string classngramfname = config.arguments[1];
    string wordpfname = config.arguments[2];
    string infname = config.arguments[3];

    bool root_unk_states = config["use-root-node"].specified;

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
    int lm_start_node = lm.advance(lm.root_node, lm.vocabulary_lookup.at("<s>"));

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

    WordClasses wcs;
    cerr << "Reading word probs.." << endl;
    wcs.read_word_probs(wordpfname);
    cerr << "Asserting class membership probabilities.." << endl;
    if (!wcs.assert_word_probs()) {
        cerr << "Problem in class membership probabilities" << endl;
        //exit(1);
    }

    cerr << "Reading class n-gram model.." << endl;
    Ngram class_ng;
    class_ng.read_arpa(classngramfname);
    int class_lm_start_node = class_ng.advance(class_ng.root_node, class_ng.vocabulary_lookup.at("<s>"));

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        if (class_ng.vocabulary_lookup.find(int2str(i)) != class_ng.vocabulary_lookup.end())
            indexmap[i] = class_ng.vocabulary_lookup[int2str(i)];
        else indexmap[i] = -1;

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

        double sent_ll = 0.0;

        vector<string> words;
        preprocess_sent(line, wcs, unk, words, num_words, num_oovs);

        int curr_class_lm_node = class_lm_start_node;
        int curr_lm_node = lm_start_node;

        for (int i=0; i<(int)words.size(); i++) {
            if (words[i] == unk) {
                if (root_unk_states) {
                    curr_lm_node = lm.root_node;
                    curr_class_lm_node = class_ng.root_node;
                }
                else {
                    curr_lm_node = lm.advance(curr_lm_node, unk_id);
                    curr_class_lm_node = class_ng.advance(curr_class_lm_node, indexmap[UNK_CLASS]);
                }
                continue;
            }

            double ngram_score = 0.0;
            curr_lm_node = lm.score(curr_lm_node, lm.vocabulary_lookup.at(words[i]), ngram_score);
            ngram_score *= log(10.0);
            ngram_score += word_iw;

            const WordClassProbs &wcp = wcs.m_class_memberships.at(words[i]);
            assert(wcp.size() == 1);
            double class_score = 0.0;
            curr_class_lm_node = class_ng.score(curr_class_lm_node, indexmap[wcp.begin()->first], class_score);
            class_score *= log(10.0);
            class_score += wcp.begin()->second;
            class_score += class_iw;

            sent_ll += add_log_domain_probs(ngram_score, class_score);
        }

        double ngram_score = 0.0;
        curr_lm_node = lm.score(curr_lm_node, lm.vocabulary_lookup.at("</s>"), ngram_score);
        ngram_score *= log(10.0);
        ngram_score += word_iw;

        double class_score = 0.0;
        curr_class_lm_node = class_ng.score(curr_class_lm_node, class_ng.vocabulary_lookup.at("</s>"), class_score);
        class_score *= log(10.0);
        class_score += class_iw;

        sent_ll += add_log_domain_probs(ngram_score, class_score);

        total_ll += sent_ll;
        num_sents++;
    }

    double ppl = exp(-1.0/double(num_words) * total_ll);
    cerr << endl;
    cerr << "Number of sentences: " << num_sents << endl;
    //cerr << "Number of skipped sentences: " << num_skipped_sents << endl;
    cerr << "Number of in-vocabulary words excluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oovs << endl;
    cerr << "Total log likelihood (ln): " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/2.302585092994046 << endl;
    cerr << "Perplexity: " << ppl << endl;

    exit(0);
}

