#include <fstream>
#include <iostream>
#include <sstream>

#include "str.hh"
#include "defs.hh"
#include "conf.hh"
#include "ModelWrappers.hh"

using namespace std;

int main(int argc, char* argv[])
{
    conf::Config config;
    config("usage: ngramppl [OPTION...] ARPAFILE INPUT\n")
            ('r', "unk-root-node", "", "",
                    "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
            ('w', "num-words=INT", "arg", "", "Number of words for computing word-normalized perplexity")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=2) config.print_help(stderr, 1);

    string arpafname = config.arguments[0];
    string infname = config.arguments[1];

    WordNgram lm(arpafname, config["unk-root-node"].specified);

    SimpleFileInput infile(infname);
    string line;
    long int num_words = 0;
    long int num_sents = 0;
    long int num_oov = 0;
    double total_ll = 0.0;
    int linei = 0;
    while (infile.getline(line)) {

        line = str::cleaned(line);
        if (line.length()==0) continue;
        if (++linei%10000==0) cerr << "sentence " << linei << endl;

        stringstream ss(line);
        vector<string> words;
        string word;
        while (ss >> word) {
            if (word==SENTENCE_BEGIN_SYMBOL) continue;
            if (word==SENTENCE_END_SYMBOL) continue;
            words.push_back(word);
        }

        double sent_ll = 0.0;
        lm.start_sentence();
        for (auto wit = words.begin(); wit!=words.end(); ++wit) {
            if (lm.word_in_vocabulary(*wit)) {
                sent_ll += lm.likelihood(*wit);
                num_words++;
            }
            else {
                lm.likelihood(*wit);
                num_oov++;
            }
        }
        sent_ll += lm.sentence_end_likelihood();
        num_words++;

        total_ll += sent_ll;
        num_sents++;
    }

    cerr << endl;
    cerr << "Number of sentences: " << num_sents << endl;
    cerr << "Number of in-vocabulary words exluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oov << endl;
    cerr << "Total log likelihood (ln): " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/2.302585092994046 << endl;

    double ppl = exp(-1.0/double(num_words) * total_ll);
    cerr << "Perplexity: " << ppl << endl;

    if (config["num-words"].specified) {
        double wnppl = exp(-1.0/double(config["num-words"].get_int()) * total_ll);
        cerr << "Word-normalized perplexity: " << wnppl << endl;
    }

    exit(EXIT_SUCCESS);
}
