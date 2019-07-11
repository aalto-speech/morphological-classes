#include "conf.hh"
#include "ModelWrappers.hh"

using namespace std;

int main(int argc, char* argv[])
{
    conf::Config config;
    config("usage: classppl [OPTION...] CLASS_ARPA CLASS_MEMBERSHIPS INPUT\n")
            ('r', "unk-root-node", "", "",
                    "Pass through root node in contexts with unks, DEFAULT: advance with unk symbol")
            ('w', "num-words=INT", "arg", "", "Number of words for computing word-normalized perplexity")
            ('f', "prob-file", "arg", "", "Write log likelihoods (ln) to a file")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=3) config.print_help(stderr, 1);

    string ngramfname = config.arguments[0];
    string classmfname = config.arguments[1];
    string infname = config.arguments[2];

    ClassNgram lm(ngramfname, classmfname, config["unk-root-node"].specified);
    int num_words = config["num-words"].specified ? config["num-words"].get_int() : 0;
    string prob_file = config["prob-file"].specified ? config["prob-file"].get_str() : "";
    lm.evaluate(
            infname,
            config["prob-file"].specified ? &prob_file : nullptr,
            config["num-words"].specified ? &num_words : nullptr);

    exit(EXIT_SUCCESS);
}
