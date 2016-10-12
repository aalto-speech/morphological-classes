#include <iostream>
#include <string>
#include <cmath>
#include <sstream>

#include "Classes.hh"
#include "conf.hh"

using namespace std;


struct TrainingParameters {
    int num_tokens;
    int num_final_tokens;
    int num_classes_per_word;
    int num_threads;
    int max_sent_length;
    flt_type prob_beam;
    bool write_temp_models;
    int top_word_classes;
    bool verbose;
};


void read_params(conf::Config &config,
                 TrainingParameters &params)
{
    params.num_tokens = config["tokens-per-word"].get_int();
    params.num_final_tokens = config["final-tokens"].get_int();
    params.num_classes_per_word = config["classes-per-word"].get_int();
    params.num_threads = config["threads"].get_int();
    params.max_sent_length = config["max-sent-length"].get_int();
    params.prob_beam = config["beam"].get_float();
    params.write_temp_models = config["write-temporary-models"].specified;
    params.top_word_classes = config["top-word-classes"].get_int();
    params.verbose = config["verbose"].specified;
}


void print_params(TrainingParameters &params)
{
    cerr << "Tokens per word: " << params.num_tokens << endl;
    cerr << "Tokens for accumulating stats: " << params.num_final_tokens << endl;
    cerr << "Max classes per word: ";
    if (params.num_classes_per_word > 0) cerr << params.num_classes_per_word << endl;
    else cerr << "NO" << endl;
    cerr << "Number of threads: " << params.num_threads << endl;
    cerr << "Maximum sentence length: " << params.max_sent_length << endl;
    cerr << "Log probability beam: " << params.prob_beam << endl;
    cerr << "Write temporary models: " << params.write_temp_models << endl;
    cerr << "An own class for the most common words: " << params.top_word_classes << endl;
    cerr << "Verbose: " << params.verbose << endl << endl;
}


void train(TrainingParameters &params,
           vector<vector<string> > &sents,
           int num_classes,
           WordClasses **wcl,
           ClassNgram **ngram,
           ClassNgram *ngram_stats,
           bool update_classes,
           string slug)
{
    WordClasses *word_stats = new WordClasses(num_classes);

    flt_type ll = collect_stats_thr(sents, *ngram, *wcl, ngram_stats, word_stats,
                                    params.num_threads, params.num_tokens,
                                    params.num_final_tokens, params.prob_beam,
                                    params.verbose);

    if (update_classes) {
        if (params.num_classes_per_word > 0)
            limit_num_classes(word_stats->m_stats, params.num_classes_per_word);
        word_stats->estimate_model();
        word_stats->assert_class_probs();
        delete *wcl;
        *wcl = word_stats;
    }
    else delete word_stats;

    ngram_stats->estimate_model();
    delete *ngram;
    *ngram = ngram_stats;

    cerr << "Total likelihood: " << ll << endl;
    cerr << "Number of n-grams: " << (*ngram)->num_grams() << endl;
    cerr << "Number of observed classes: " << (*wcl)->num_observed_classes() << endl;
    cerr << "Number of class probabilities: " << (*wcl)->num_class_probs() << endl;
    if (params.write_temp_models) {
        (*ngram)->write_model(slug + ".ngram.gz");
        (*wcl)->write_class_probs(slug + ".cgenprobs.gz");
        (*wcl)->write_word_probs(slug + ".cmemprobs.gz");
    }
}


int main(int argc, char* argv[])
{
    conf::Config config;
    config("usage: train [OPTION...] INIT_WORDS CORPUS MODEL\n")
    ('o', "tokens-per-word=INT", "arg", "100", "Number of top tokens to propagate in each word position")
    ('f', "final-tokens=INT", "arg", "10", "Number of top tokens for accumulating statistics")
    ('b', "beam=FLOAT", "arg", "20.0", "Log probability pruning beam for class paths, default 20.0")
    ('c', "classes-per-word=INT", "arg", "0", "Maximum number of classes per word")
    ('w', "top-word-classes=INT", "arg", "0", "Assign an own class for the most common words")
    ('t', "threads=INT", "arg", "1", "Number of threads for collecting statistics")
    ('l', "max-sent-length=INT", "arg", "20", "Maximum length for training sentences")
    ('i', "num-iterations=INT", "arg", "10", "Number of trigram training iterations")
    ('m', "write-temporary-models", "", "", "Writes models after each iteration")
    ('v', "verbose", "", "", "Print some extra information")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 3) config.print_help(stderr, 1);

    try {
        string init_words_fname = config.arguments[0];
        string corpus_fname = config.arguments[1];
        string model_fname = config.arguments[2];

        TrainingParameters params;
        read_params(config, params);
        cerr << std::boolalpha;
        print_params(params);

        map<string, int> word_counts;
        int wc = get_word_counts(corpus_fname, word_counts);
        WordClasses *wcl = new WordClasses(init_words_fname, word_counts, params.top_word_classes);
        word_counts.clear();
        wcl->assert_class_probs();
        int num_classes = wcl->num_classes();

        cerr << "Read class probabilities for " << wcl->num_words() << " words" << endl;
        cerr << "Number of words with classes: " << wcl->num_words_with_classes() << endl;
        cerr << "Number of classes: " << num_classes << endl;
        cerr << "Number of class probabilities: " << wcl->num_class_probs() << endl;
        cerr << "Number of word probabilities: " << wcl->num_word_probs() << endl;

        vector<vector<string> > sents;
        int num_unk_tokens, num_unk_types, num_word_tokens, num_word_types;
        set<string> vocab_wo_unanalyzed; wcl->get_words(vocab_wo_unanalyzed, false);
        set<string> vocab_with_unanalyzed; wcl->get_words(vocab_with_unanalyzed, true);
        read_sents(corpus_fname, sents, params.max_sent_length,
                   &vocab_wo_unanalyzed,
                   &num_word_tokens, &num_word_types,
                   &num_unk_tokens, &num_unk_types);
        vocab_wo_unanalyzed.clear();

        cerr << endl;
        cerr << "Training corpus sentences: " << sents.size() << endl;
        cerr << "Training corpus word tokens: " << num_word_tokens << endl;
        cerr << "Training corpus word types: " << num_word_types << endl;
        cerr << "Training corpus unk tokens: " << num_unk_tokens << endl;
        cerr << "Training corpus unk types: " << num_unk_types << endl;

        cerr << endl << "Unigram iteration 0" << endl;
        ClassNgram *ngram = new Unigram(num_classes);
        train(params, sents, num_classes, &wcl, &ngram, new Unigram(), false, "unigram.iter0");

        cerr << endl << "Bigram iteration 0" << endl;
        train(params, sents, num_classes, &wcl, &ngram, new Bigram(), false, "bigram.iter0");

        cerr << endl << "Reloading corpus with unanalyzed words included" << endl;
        read_sents(corpus_fname, sents, params.max_sent_length,
                   &vocab_with_unanalyzed,
                   &num_word_tokens, &num_word_types,
                   &num_unk_tokens, &num_unk_types);
        vocab_with_unanalyzed.clear();
        cerr << "Training corpus sentences: " << sents.size() << endl;
        cerr << "Training corpus word tokens: " << num_word_tokens << endl;
        cerr << "Training corpus word types: " << num_word_types << endl;
        cerr << "Training corpus unk tokens: " << num_unk_tokens << endl;
        cerr << "Training corpus unk types: " << num_unk_types << endl;

        for (int i=0; i<config["num-iterations"].get_int(); i++) {
            cerr << endl << "Trigram iteration " << i << endl;
            train(params, sents, num_classes, &wcl, &ngram, new Trigram(), true, "trigram.iter"+int2str(i));
        }

        ngram->write_model(model_fname + ".ngram.gz");
        wcl->write_class_probs(model_fname + ".cgenprobs.gz");
        wcl->write_word_probs(model_fname + ".cmemprobs.gz");

    } catch (string &e) {
        cerr << e << endl;
    }

    exit(0);
}
