#include <fstream>
#include <iostream>
#include <sstream>

#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Classes.hh"

using namespace std;


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catppl [OPTION...] CLASS_NGRAM CLASS_PROBS WORD_PROBS INPUT\n")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 4) config.print_help(stderr, 1);

    string ngramfname = config.arguments[0];
    string classpfname = config.arguments[1];
    string wordpfname = config.arguments[2];
    string infname = config.arguments[3];

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

    cerr << "Reading class trigram model.." << endl;
    Trigram tg;
    tg.read_model(ngramfname);

    cerr << "Reading input.." << endl;
    SimpleFileInput infile(infname);
    string line;
    string start_symbol = "<s>";
    string end_symbol = "<s>";
    long int num_words = 0;
    long int num_sents = 0;
    long int num_oov = 0;
    double total_ll = 0.0;
    int linei = 0;
    while (infile.getline(line)) {

        linei++;
        if (linei % 10000 == 0) cerr << "sentence " << linei << endl;

        stringstream ss(line);
        vector<string> words = { start_symbol, start_symbol };
        string word;
        while (ss >> word) {
            if (word == "<s>") continue;
            if (word == "</s>") continue;
            if (wcs.m_class_gen_probs.find(word) == wcs.m_class_gen_probs.end()) {
                words.push_back("<unk>");
                num_oov++;
            }
            else {
                words.push_back(word);
                num_words++;
            }
        }
        words.push_back(end_symbol);
        num_words++;

        double sent_ll = 0.0;
        for (int w=2; w<(int)words.size(); w++) {
            string &c2w = words[w-2];
            string &c1w = words[w-1];
            string &word = words[w];
            if (word == "<unk>") continue;

            double word_ll = MIN_LOG_PROB;
            for (auto c2it=wcs.m_class_gen_probs[c2w].begin(); c2it != wcs.m_class_gen_probs[c2w].end(); ++c2it) {
                auto ng2it = tg.m_trigrams.find(c2it->first);
                if (ng2it == tg.m_trigrams.end()) continue;
                for (auto c1it=wcs.m_class_gen_probs[c1w].begin(); c1it != wcs.m_class_gen_probs[c1w].end(); ++c1it) {
                    auto ng1it = ng2it->second.find(c1it->first);
                    if (ng1it == ng2it->second.end()) continue;
                    for (auto cwit=wcs.m_class_memberships[word].begin(); cwit != wcs.m_class_memberships[word].end(); ++cwit) {
                        auto ngit = ng1it->second.find(cwit->first);
                        if (ngit == ng1it->second.end()) continue;
                        double prob = ngit->second;
                        prob += c2it->second;
                        prob += c1it->second;
                        prob += cwit->second;
                        word_ll = add_log_domain_probs(word_ll, prob);
                    }
                }
            }

            //if (word_ll == MIN_LOG_PROB) num_words--;
            //else sent_ll += word_ll;
            sent_ll += word_ll;

        }

        total_ll += sent_ll;
        num_sents++;
    }

    double ppl = exp(-1.0/double(num_words) * total_ll);
    cerr << endl;
    cerr << "Number of sentences: " << num_sents << endl;
    cerr << "Number of in-vocabulary words exluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oov << endl;
    cerr << "Total log likelihood (ln): " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/2.302585092994046 << endl;
    cerr << "Perplexity: " << ppl << endl;

    exit(0);
}

