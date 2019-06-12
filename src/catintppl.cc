#include <sstream>

#include "str.hh"
#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "ModelWrappers.hh"

using namespace std;

int main(int argc, char* argv[])
{

    conf::Config config;
    config("usage: catintppl [OPTION...] WORD_ARPA CAT_ARPA CGENPROBS CMEMPROBS INPUT\n")
            ('i', "weight=FLOAT", "arg", "0.5", "Interpolation weight [0.0,1.0] for the word ARPA model")
            ('r', "unk-root-node", "", "",
                    "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
            ('n', "num-tokens=INT", "arg", "100",
                    "Upper limit for the number of tokens in each position (DEFAULT: 100)")
            ('b', "prob-beam=FLOAT", "arg", "20.0", "Probability beam (DEFAULT: 20.0)")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=5)
        config.print_help(stderr, 1);

    string arpafname = config.arguments[0];
    string cngramfname = config.arguments[1];
    string cgenpfname = config.arguments[2];
    string cmempfname = config.arguments[3];
    string infname = config.arguments[4];

    bool root_unk_states = config["unk-root-node"].specified;
    int max_tokens = config["num-tokens"].get_int();
    double prob_beam = config["prob-beam"].get_float();

    double iw = config["weight"].get_float();
    if (iw<0.0 || iw>1.0) {
        cerr << "Invalid interpolation weight: " << iw << endl;
        exit(EXIT_FAILURE);
    }
    cerr << "Interpolation weight: " << iw << endl;
    double word_iw = log(iw);
    double cat_iw = log(1.0-iw);

    WordNgram wlm(arpafname, config["unk-root-node"].specified);
    CategoryNgram clm(
            cngramfname, cgenpfname, cmempfname,
            config["unk-root-node"].specified,
            config["num-tokens"].get_int(),
            config["prob-beam"].get_float());

    SimpleFileInput infile(infname);
    string line;
    long int num_words = 0;
    long int num_sents = 0;
    long int num_oovs = 0;
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
        wlm.start_sentence();
        clm.start_sentence();
        for (auto wit = words.begin(); wit!=words.end(); ++wit) {
            if (wlm.word_in_vocabulary(*wit) && clm.word_in_vocabulary(*wit)) {
                sent_ll += add_log_domain_probs(
                        word_iw+wlm.likelihood(*wit),
                        cat_iw+clm.likelihood(*wit));
                num_words++;
            }
            else {
                wlm.likelihood(UNK_SYMBOL);
                clm.likelihood(UNK_SYMBOL);
                num_oovs++;
            }
        }
        sent_ll += add_log_domain_probs(
                word_iw+wlm.sentence_end_likelihood(),
                cat_iw+clm.sentence_end_likelihood());
        num_words++;

        total_ll += sent_ll;
        num_sents++;
    }

    cerr << endl;
    cerr << "Number of sentences: " << num_sents << endl;
    cerr << "Number of in-vocabulary words excluding sentence ends: " << num_words-num_sents << endl;
    cerr << "Number of in-vocabulary words including sentence ends: " << num_words << endl;
    cerr << "Number of OOV words: " << num_oovs << endl;
    cerr << "Total log likelihood (ln): " << total_ll << endl;
    cerr << "Total log likelihood (log10): " << total_ll/2.302585092994046 << endl;

    double ppl = exp(-1.0/double(num_words) * total_ll);
    cerr << "Perplexity: " << ppl << endl;


    exit(EXIT_SUCCESS);
}
