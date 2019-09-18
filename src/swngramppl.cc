#include "conf.hh"
#include "ModelWrappers.hh"

using namespace std;

int main(int argc, char* argv[])
{
    conf::Config config;
    config("usage: swngramppl [OPTION...] ARPAFILE WORD_SEGMENTATIONS INPUT\n")
            ('f', "prob-file", "arg", "", "Write log likelihoods (ln) to a file")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=3) config.print_help(stderr, 1);

    string arpa_fname = config.arguments[0];
    string word_segs_fname = config.arguments[1];
    string infname = config.arguments[2];

    try {
        SubwordNgram lm(arpa_fname, word_segs_fname);
        string prob_file = config["prob-file"].specified ? config["prob-file"].get_str() : "";
        lm.evaluate(
                infname,
                config["prob-file"].specified ? &prob_file : nullptr,
                nullptr);
    } catch (string e) {
        cerr << e << endl;
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}
