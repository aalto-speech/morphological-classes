#include "conf.hh"
#include "ModelWrappers.hh"

using namespace std;

int main(int argc, char* argv[])
{
    conf::Config config;
    config("usage: classintppl [OPTION...] ARPAFILE CLASS_ARPA CLASS_MEMBERSHIPS INPUT\n")
            ('i', "weight=FLOAT", "arg", "0.5", "Interpolation weight [0.0,1.0] for the word ARPA model")
            ('r', "unk-root-node", "", "",
                    "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
            ('w', "num-words=INT", "arg", "", "Number of words for computing word-normalized perplexity")
            ('f', "prob-file", "arg", "", "Write log likelihoods (ln) to a file")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=4) config.print_help(stderr, 1);

    string arpa_fname = config.arguments[0];
    string class_ngram_fname = config.arguments[1];
    string class_m_fname = config.arguments[2];
    string infname = config.arguments[3];

    WordNgram wlm(arpa_fname, config["unk-root-node"].specified);
    ClassNgram clm(class_ngram_fname, class_m_fname, config["unk-root-node"].specified);
    InterpolatedLM lm(&wlm, &clm, config["weight"].get_float());

    int num_words = config["num-words"].specified ? config["num-words"].get_int() : 0;
    string prob_file = config["prob-file"].specified ? config["prob-file"].get_str() : "";
    lm.evaluate(
            infname,
            config["prob-file"].specified ? &prob_file : nullptr,
            config["num-words"].specified ? &num_words : nullptr);

    exit(EXIT_SUCCESS);
}
