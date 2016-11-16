#include <string>
#include <cmath>
#include <sstream>

#include "Categories.hh"
#include "conf.hh"

using namespace std;


int
get_word_counts(string corpusfname,
                map<string, int> &counts)
{
    SimpleFileInput corpusf(corpusfname);

    int wc = 0;
    string line;
    while (corpusf.getline(line)) {
        stringstream ss(line);
        string word;
        while (ss >> word) {
            if (word == "<s>" || word == "</s>") continue;
            counts[word]++;
            wc++;
        }
    }

    return wc;
}


int main(int argc, char* argv[])
{
    conf::Config config;
    config("usage: init [OPTION...] INIT_WORDS CORPUS MODEL\n")
    ('w', "top-word-classes=INT", "arg", "0", "Assign an own class for the most common words")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 3) config.print_help(stderr, 1);

    try {
        string init_words_fname = config.arguments[0];
        string corpus_fname = config.arguments[1];
        string model_fname = config.arguments[2];
        int top_word_classes = config["top-word-classes"].get_int();

        map<string, int> word_counts;
        get_word_counts(corpus_fname, word_counts);
        Categories *wcl = new Categories(init_words_fname, word_counts, top_word_classes);
        wcl->assert_category_gen_probs();
        wcl->assert_category_mem_probs();
        int num_classes = wcl->num_classes();

        cerr << "Read class probabilities for " << wcl->num_words() << " words" << endl;
        cerr << "Number of words with categories: " << wcl->num_words_with_categories() << endl;
        cerr << "Number of categories: " << num_classes << endl;
        cerr << "Number of category generation probabilities: " << wcl->num_category_gen_probs() << endl;
        cerr << "Number of category membership probabilities: " << wcl->num_category_mem_probs() << endl;

        wcl->write_category_gen_probs(model_fname + ".cgenprobs.gz");
        wcl->write_category_mem_probs(model_fname + ".cmemprobs.gz");

    } catch (string &e) {
        cerr << e << endl;
    }

    exit(0);
}
