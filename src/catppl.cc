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
    config("usage: catppl [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT\n")
            ('r', "unk-root-node", "", "",
                    "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
            ('n', "num-tokens=INT", "arg", "100",
                    "Upper limit for the number of tokens in each position (DEFAULT: 100)")
            ('b', "prob-beam=FLOAT", "arg", "20.0", "Probability beam (DEFAULT: 20.0)")
            ('w', "num-words=INT", "arg", "", "Number of words for computing word-normalized perplexity")
            ('f', "prob-file", "arg", "", "Write log likelihoods (ln) to a file")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=4)
        config.print_help(stderr, 1);

    string cngramfname = config.arguments[0];
    string cgenpfname = config.arguments[1];
    string cmempfname = config.arguments[2];
    string infname = config.arguments[3];

    CategoryNgram lm(
            cngramfname, cgenpfname, cmempfname,
            config["unk-root-node"].specified,
            config["num-tokens"].get_int(),
            config["prob-beam"].get_float());

    int num_words = config["num-words"].specified ? config["num-words"].get_int() : 0;
    string prob_file = config["prob-file"].specified ? config["prob-file"].get_str() : "";
    lm.evaluate(
            infname,
            config["prob-file"].specified ? &prob_file : nullptr,
            config["num-words"].specified ? &num_words : nullptr);

    exit(EXIT_SUCCESS);
}
