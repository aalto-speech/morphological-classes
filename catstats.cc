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
    ('p', "num-parses=INT", "arg", "10", "Maximum number of parses to print per sentence (DEFAULT: 10)")
    ('t', "num-tokens=INT", "arg", "100", "Upper limit for the number of tokens in each position (DEFAULT: 100)")
    ('e', "num-end-tokens=INT", "arg", "10", "Upper limit for the number of tokens in the end position (DEFAULT: 10)")
    ('l', "max-line-length=INT", "arg", "100", "Maximum sentence length as number of words (DEFAULT: 100)")
    ('b', "prob-beam=FLOAT", "arg", "100.0", "Probability beam (default 100.0)")
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

    if (num_parses > num_end_tokens) {
        cerr << "Warning, num-parses higher than num-end-tokens" << endl;
        cerr << "num-parses set to: " << num_end_tokens << endl;
        num_parses = num_end_tokens;
    }

    Categories wcs;
    cerr << "Reading category generation probs.." << endl;
    wcs.read_category_gen_probs(cgenpfname);
    cerr << "Reading category membership probs.." << endl;
    wcs.read_category_mem_probs(cmempfname);

    cerr << "Reading category n-gram model.." << endl;
    Ngram cngram;
    cngram.read_arpa(cngramfname);

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        if (cngram.vocabulary_lookup.find(int2str(i)) != cngram.vocabulary_lookup.end())
            indexmap[i] = cngram.vocabulary_lookup[int2str(i)];

    set<string> vocab; wcs.get_words(vocab, false);

    Categories stats;

    SimpleFileInput corpusf(infname);
    SimpleFileOutput outf(outfname);
    string line;
    int senti=0;
    flt_type total_ll = 0.0;
    while (corpusf.getline(line)) {
        vector<string> sent;
        stringstream ss(line);
        string word;
        while (ss >> word) {
            if (word == "<s>" || word == "</s>") continue;
            sent.push_back(word);
        }
        if ((int)sent.size() > max_line_length) continue;
        if ((int)sent.size() == 0) continue;

        for (auto wit=sent.begin(); wit != sent.end(); ++wit)
            if (vocab.find(*wit) == vocab.end())
                wit->assign("<unk>");

        flt_type ll = collect_stats(sent,
                                    cngram, indexmap,
                                    wcs,
                                    stats, outf,
                                    num_tokens, num_end_tokens,
                                    num_parses, prob_beam, false);
        total_ll += ll;
        if (++senti % 10000 == 0) cerr << "Processing sentence " << senti << endl;
    }

    outf.close();

    stats.estimate_model();
    stats.write_category_gen_probs(modelfname + ".cgenprobs.gz");
    stats.write_category_mem_probs(modelfname + ".cmemprobs.gz");

    cerr << "Number of sentences processed: " << senti << endl;
    cerr << "Likelihood: " << total_ll << endl;

    exit(0);
}

