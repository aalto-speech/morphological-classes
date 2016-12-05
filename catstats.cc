#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <functional>

#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Categories.hh"
#include "Ngram.hh"

using namespace std;


bool
process_sent(string line,
             const set<string> &vocab,
             const TrainingParameters &params,
             vector<string> &sent)
{
    sent.clear();
    stringstream ss(line);
    string word;
    while (ss >> word) {
        if (word == "<s>" || word == "</s>") continue;
        sent.push_back(word);
    }
    if (sent.size() > params.max_line_length) return false;
    if (sent.size() == 0) return false;

    for (auto wit=sent.begin(); wit != sent.end(); ++wit)
        if (vocab.find(*wit) == vocab.end())
            wit->assign("<unk>");

    return true;
}


void
catstats(string corpusfname,
         const set<string> &vocab,
         const Ngram &cngram,
         const vector<int> &indexmap,
         const Categories &categories,
         const TrainingParameters &params,
         Categories &stats,
         string modelfname,
         unsigned long int &num_vocab_words,
         unsigned long int &num_oov_words,
         unsigned long int &num_sents,
         flt_type &total_ll,
         unsigned int num_threads=1,
         unsigned int thread_idx=0)
{
    SimpleFileOutput *seqf = nullptr;
    if (modelfname.length() > 0)
        seqf = new SimpleFileOutput(modelfname + ".catseq.gz");

    SimpleFileInput corpusf(corpusfname);
    string line;
    unsigned long int senti=0;
    while (corpusf.getline(line)) {
        senti++;
        if (senti % num_threads != thread_idx) continue;
        vector<string> sent;
        if (!process_sent(line, vocab, params, sent)) continue;
        total_ll += collect_stats(sent,
                                  cngram, indexmap,
                                  categories, params,
                                  stats, seqf,
                                  &num_vocab_words, &num_oov_words);
        num_sents++;
    }

    if (seqf != nullptr) {
        seqf->close();
        delete seqf;
    }
}


flt_type
catstats_thr(string corpusfname,
             const set<string> &vocab,
             const Ngram &cngram,
             const vector<int> &indexmap,
             const Categories &categories,
             const TrainingParameters &params,
             Categories &stats,
             string modelfname,
             unsigned long int &num_vocab_words,
             unsigned long int &num_oov_words,
             unsigned long int &num_sents,
             unsigned int num_threads)
{
    vector<unsigned long int> thr_num_vocab_words(num_threads, 0);
    vector<unsigned long int> thr_num_oov_words(num_threads, 0);
    vector<unsigned long int> thr_num_sents(num_threads, 0);
    vector<flt_type> thr_ll(num_threads, 0.0);
    vector<Categories*> thr_stats(num_threads, nullptr);
    vector<std::thread*> workers;
    for (unsigned int t=0; t<num_threads; t++) {
        thr_stats[t] = new Categories(categories.num_categories());
        std::thread *worker = new std::thread(&catstats,
                                              corpusfname,
                                              std::cref(vocab),
                                              std::cref(cngram),
                                              std::cref(indexmap),
                                              std::cref(categories),
                                              std::cref(params),
                                              std::ref(*(thr_stats[t])),
                                              modelfname + ".thread" + int2str(t),
                                              std::ref(thr_num_vocab_words[t]),
                                              std::ref(thr_num_oov_words[t]),
                                              std::ref(thr_num_sents[t]),
                                              std::ref(thr_ll[t]),
                                              num_threads,
                                              t);
        workers.push_back(worker);
    }
    flt_type total_ll=0.0;
    for (unsigned int t=0; t<num_threads; t++) {
        workers[t]->join();
        stats.accumulate(*(thr_stats[t]));
        delete thr_stats[t];
        num_vocab_words += thr_num_vocab_words[t];
        num_oov_words += thr_num_oov_words[t];
        num_sents += thr_num_sents[t];
        total_ll += thr_ll[t];
        delete workers[t];
    }
    return total_ll;
}


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catstats [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT [MODEL]\n")
    ('p', "num-parses=INT", "arg", "10", "Maximum number of parses to print per sentence (DEFAULT: 10)")
    ('n', "num-tokens=INT", "arg", "100", "Upper limit for the number of tokens in each position (DEFAULT: 100)")
    ('f', "num-final-tokens=INT", "arg", "10", "Upper limit for the number of tokens in the last position (DEFAULT: 10)")
    ('l', "max-line-length=INT", "arg", "100", "Maximum sentence length as number of words (DEFAULT: 100)")
    ('o', "max-order=INT", "arg", "", "Maximum context length (DEFAULT: MODEL ORDER)")
    ('b', "prob-beam=FLOAT", "arg", "100.0", "Probability beam (default 100.0)")
    ('u', "update-categories", "", "", "Update category generation and membership probabilities")
    ('t', "num-threads=INT", "arg", "1", "Number of threads")
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

    TrainingParameters params;
    params.num_parses = config["num-parses"].get_int();
    params.num_tokens = config["num-tokens"].get_int();
    params.num_final_tokens = config["num-final-tokens"].get_int();
    params.max_line_length = config["max-line-length"].get_int();
    params.prob_beam = config["prob-beam"].get_float();
    bool update_categories = config["update-categories"].specified;

    if (params.num_parses > params.num_final_tokens) {
        cerr << "Warning, num-parses higher than num-final-tokens" << endl;
        cerr << "num-parses set to: " << params.num_final_tokens << endl;
        params.num_parses = params.num_final_tokens;
    }

    Categories wcs;
    cerr << "Reading category generation probs.." << endl;
    wcs.read_category_gen_probs(cgenpfname);
    cerr << "Reading category membership probs.." << endl;
    wcs.read_category_mem_probs(cmempfname);

    cerr << "Reading category n-gram model.." << endl;
    Ngram cngram;
    cngram.read_arpa(cngramfname);
    params.max_order = cngram.max_order;
    if (config["max-order"].specified) params.max_order = config["max-order"].get_int();

    // The class indexes are stored as strings in the n-gram class
    vector<int> indexmap(wcs.num_categories());
    for (int i=0; i<(int)indexmap.size(); i++)
        if (cngram.vocabulary_lookup.find(int2str(i)) != cngram.vocabulary_lookup.end())
            indexmap[i] = cngram.vocabulary_lookup[int2str(i)];

    set<string> vocab; wcs.get_words(vocab, false);

    Categories stats(wcs.num_categories());

    time_t now = time(0);
    cerr << std::ctime(&now) << endl;

    unsigned long int num_vocab_words=0;
    unsigned long int num_oov_words=0;
    unsigned long int num_sents=0;
    flt_type total_ll=0.0;
    if (config["num-threads"].get_int() > 1)
        total_ll = catstats_thr(infname, vocab,
                     cngram, indexmap, wcs,
                     params,
                     stats, modelfname,
                     num_vocab_words, num_oov_words, num_sents,
                     config["num-threads"].get_int());

    else
        catstats(infname, vocab,
                 cngram, indexmap, wcs,
                 params,
                 stats, modelfname,
                 num_vocab_words, num_oov_words, num_sents, total_ll);

    now = time(0);
    cerr << std::ctime(&now) << endl;

    cout << "Number of sentences processed: " << num_sents << endl;
    cout << "Number of in-vocabulary word tokens without sentence ends: " << num_vocab_words << endl;
    cout << "Number of in-vocabulary word tokens with sentence ends: " << num_vocab_words+num_sents << endl;
    cout << "Number of out-of-vocabulary word tokens: " << num_oov_words << endl;
    cout << "Likelihood: " << total_ll << endl;
    double ppl = exp(-1.0/double(num_vocab_words+num_sents) * total_ll);
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

