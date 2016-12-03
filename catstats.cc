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


flt_type catstats(SimpleFileInput &corpusf,
                  set<string> &vocab,
                  const Ngram &cngram,
                  const vector<int> &indexmap,
                  const Categories &categories,
                  Categories &stats,
                  SimpleFileOutput *seqf,
                  unsigned int num_tokens,
                  unsigned int num_final_tokens,
                  unsigned int num_parses,
                  unsigned int max_order,
                  unsigned int max_line_length,
                  flt_type prob_beam,
                  bool verbose,
                  unsigned long int &num_vocab_words,
                  unsigned long int &num_oov_words,
                  unsigned long int &senti)
{
    string line;
    flt_type total_ll = 0.0;
    while (corpusf.getline(line)) {
        vector<string> sent;
        stringstream ss(line);
        string word;
        while (ss >> word) {
            if (word == "<s>" || word == "</s>") continue;
            sent.push_back(word);
        }
        if (sent.size() > max_line_length) continue;
        if (sent.size() == 0) continue;

        for (auto wit=sent.begin(); wit != sent.end(); ++wit)
            if (vocab.find(*wit) == vocab.end())
                wit->assign("<unk>");

        flt_type ll = collect_stats(sent,
                                    cngram, indexmap,
                                    categories,
                                    stats, seqf,
                                    num_tokens, num_final_tokens,
                                    num_parses, max_order,
                                    prob_beam, false,
                                    &num_vocab_words, &num_oov_words);
        total_ll += ll;
        if (++senti % 10000 == 0) cerr << "Processing sentence " << senti << endl;
    }

    return total_ll;
}



int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catstats [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT [MODEL]\n")
    ('p', "num-parses=INT", "arg", "10", "Maximum number of parses to print per sentence (DEFAULT: 10)")
    ('t', "num-tokens=INT", "arg", "100", "Upper limit for the number of tokens in each position (DEFAULT: 100)")
    ('e', "num-end-tokens=INT", "arg", "10", "Upper limit for the number of tokens in the end position (DEFAULT: 10)")
    ('l', "max-line-length=INT", "arg", "100", "Maximum sentence length as number of words (DEFAULT: 100)")
    ('o', "max-order=INT", "arg", "", "Maximum context length (DEFAULT: MODEL ORDER)")
    ('b', "prob-beam=FLOAT", "arg", "100.0", "Probability beam (default 100.0)")
    ('u', "update-categories", "", "", "Update category generation and membership probabilities")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 4 && config.arguments.size() != 5)
        config.print_help(stderr, 1);

    string cngramfname = config.arguments[0];
    string cgenpfname = config.arguments[1];
    string cmempfname = config.arguments[2];
    string infname = config.arguments[3];
    string modelfname = "";
    if (config.arguments.size() == 5)
        modelfname = config.arguments[4];

    int num_parses = config["num-parses"].get_int();
    int num_tokens = config["num-tokens"].get_int();
    int num_end_tokens = config["num-end-tokens"].get_int();
    int max_line_length = config["max-line-length"].get_int();
    flt_type prob_beam = config["prob-beam"].get_float();
    bool update_categories = config["update-categories"].specified;

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
    int max_order = cngram.max_order;
    if (config["max-order"].specified) max_order = config["max-order"].get_int();

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_categories());
    for (int i=0; i<(int)indexmap.size(); i++)
        if (cngram.vocabulary_lookup.find(int2str(i)) != cngram.vocabulary_lookup.end())
            indexmap[i] = cngram.vocabulary_lookup[int2str(i)];

    set<string> vocab; wcs.get_words(vocab, false);

    Categories stats(wcs.num_categories());

    SimpleFileInput corpusf(infname);
    SimpleFileOutput *seqf = nullptr;
    if (modelfname.length() > 0)
        seqf = new SimpleFileOutput(modelfname + ".catseq.gz");
    unsigned long int num_vocab_words=0;
    unsigned long int num_oov_words=0;
    unsigned long int senti=0;

    flt_type total_ll = catstats(corpusf, vocab,
                                 cngram, indexmap, wcs,
                                 stats, seqf,
                                 num_tokens, num_end_tokens, num_parses,
                                 max_order, max_line_length, prob_beam,
                                 false,
                                 num_vocab_words, num_oov_words, senti);

    if (seqf != nullptr) {
        seqf->close();
        delete seqf;
    }

    cout << "Number of sentences processed: " << senti << endl;
    cout << "Number of in-vocabulary word tokens without sentence ends: " << num_vocab_words << endl;
    cout << "Number of in-vocabulary word tokens with sentence ends: " << num_vocab_words+senti << endl;
    cout << "Number of out-of-vocabulary word tokens: " << num_oov_words << endl;
    cout << "Likelihood: " << total_ll << endl;
    double ppl = exp(-1.0/double(num_vocab_words+senti) * total_ll);
    cout << "Perplexity: " << ppl << endl;

    if (modelfname.length() == 0) exit(EXIT_SUCCESS);

    if (update_categories) {
        stats.estimate_model();
        stats.write_category_gen_probs(modelfname + ".cgenprobs.gz");
        stats.write_category_mem_probs(modelfname + ".cmemprobs.gz");
    }
    else {
        wcs.write_category_gen_probs(modelfname + ".cgenprobs.gz");
        wcs.write_category_mem_probs(modelfname + ".cmemprobs.gz");
    }

    exit(EXIT_SUCCESS);
}

