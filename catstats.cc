#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Categories.hh"
#include "Ngram.hh"

using namespace std;


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catstats [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT OUTPUT MODEL\n")
    ('p', "num-parses=INT", "arg", "10", "Maximum number of parses to print per sentence")
    ('t', "num-tokens=INT", "arg", "100", "Upper limit for the number of tokens in each position")
    ('e', "num-end-tokens=INT", "arg", "10", "Upper limit for the number of tokens in the end position")
    ('l', "max-line-length=INT", "arg", "100", "Maximum sentence length as number of words")
    ('b', "prob-beam=FLOAT", "arg", "100", "Maximum sentence length as number of words")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 6) config.print_help(stderr, 1);

    string cngramfname = config.arguments[0];
    string cgenpfname = config.arguments[1];
    string cmempfname = config.arguments[2];
    string infname = config.arguments[3];
    string outfname = config.arguments[4];
    string modelfname = config.arguments[5];

    int num_parses = config["num-parses"].get_int();
    int num_tokens = config["num-tokens"].get_int();
    int num_end_tokens = config["num-end-tokens"].get_int();
    int max_line_length = config["max-line-length"].get_int();
    flt_type prob_beam = config["prob-beam"].get_float();

    Categories wcs;
    cerr << "Reading class generation probs.." << endl;
    wcs.read_category_gen_probs(cgenpfname);
    cerr << "Reading class membership probs.." << endl;
    wcs.read_category_mem_probs(cmempfname);

    cerr << "Reading class n-gram model.." << endl;
    Ngram cngram;
    cngram.read_arpa(cngramfname);

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        if (cngram.vocabulary_lookup.find(int2str(i)) != cngram.vocabulary_lookup.end())
            indexmap[i] = cngram.vocabulary_lookup[int2str(i)];
    if (indexmap[START_CLASS] == -1) {
        cerr << "warning, start class not in the model" << endl;
        exit(EXIT_FAILURE);
    }
    if (indexmap[UNK_CLASS] == -1) {
        cerr << "warning, unk class not in the model" << endl;
        exit(EXIT_FAILURE);
    }

    set<string> vocab; wcs.get_words(vocab, false);

    Categories stats;

    SimpleFileInput corpusf(infname);
    SimpleFileOutput outf(outfname);
    string line;
    int senti=1;
    flt_type total_ll = 0.0;
    while (corpusf.getline(line)) {
        vector<string> sent;
        stringstream ss(line);
        string word;
        while (ss >> word) {
            if (word == "<s>" || word == "</s>") continue;
            sent.push_back(word);
        }
        sent.push_back("</s>");
        if ((int)sent.size() > max_line_length) continue;
        if ((int)sent.size() == 1) continue;

        for (auto wit=sent.begin(); wit != sent.end(); ++wit)
            if (vocab.find(*wit) == vocab.end())
                wit->assign("<unk>");

        flt_type ll = collect_stats(sent,
                                    cngram, wcs,
                                    stats, outf,
                                    num_tokens, num_end_tokens,
                                    num_parses, prob_beam, false);
        total_ll += ll;
        senti++;
    }

    outf.close();

    stats.estimate_model();
    stats.write_category_gen_probs(modelfname + ".cgenprobs.gz");
    stats.write_category_mem_probs(modelfname + ".cmemprobs.gz");

    cerr << "Likelihood: " << total_ll << endl;

    exit(0);
}

