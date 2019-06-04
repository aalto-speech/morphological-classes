#include <string>
#include <cmath>
#include <sstream>

#include "Categories.hh"
#include "conf.hh"

using namespace std;

void
write_class_unigram_counts(const map<string, int>& word_counts,
        const Categories& wcl,
        string countfname)
{
    map<string, flt_type> category_counts;
    for (auto wit = word_counts.cbegin(); wit!=word_counts.cend(); ++wit) {
        if (wit->first==SENTENCE_BEGIN_SYMBOL || wit->first==SENTENCE_END_SYMBOL
                || wit->first==UNK_SYMBOL) {
            category_counts[wit->first] += wit->second;
            continue;
        }
        else if (wit->first==CAP_UNK_SYMBOL) {
            category_counts[UNK_SYMBOL] += wit->second;
            continue;
        }
        if (wcl.m_category_gen_probs.find(wit->first)==wcl.m_category_gen_probs.end())
            continue;
        const CategoryProbs& cprobs = wcl.m_category_gen_probs.at(wit->first);
        if (cprobs.size()==0)
            category_counts[UNK_SYMBOL] += wit->second;
        else {
            flt_type tmp = 1.0/(flt_type) cprobs.size();
            for (auto catit = cprobs.cbegin(); catit!=cprobs.end(); ++catit)
                category_counts[int2str(catit->first)] += wit->second*tmp;
        }
    }

    SimpleFileOutput countf(countfname);
    for (auto cit = category_counts.begin(); cit!=category_counts.end(); ++cit)
        countf << cit->first << "\t" << cit->second << "\n";
    countf.close();
}

int main(int argc, char* argv[])
{
    conf::Config config;
    config("usage: init [OPTION...] INIT_WORDS CORPUS MODEL\n")
            ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size()!=3) config.print_help(stderr, 1);

    try {
        string init_words_fname = config.arguments[0];
        string corpus_fname = config.arguments[1];
        string model_fname = config.arguments[2];

        map<string, int> word_counts;
        get_word_counts(corpus_fname, word_counts);
        Categories wcl(init_words_fname, word_counts);
        wcl.assert_category_gen_probs();
        wcl.assert_category_mem_probs();
        int num_classes = wcl.num_categories();

        cerr << "Read class probabilities for " << wcl.num_words() << " words" << endl;
        cerr << "Number of words with categories: " << wcl.num_words_with_categories() << endl;
        cerr << "Number of categories: " << num_classes << endl;
        cerr << "Number of category generation probabilities: " << wcl.num_category_gen_probs() << endl;
        cerr << "Number of category membership probabilities: " << wcl.num_category_mem_probs() << endl;

        wcl.write_category_gen_probs(model_fname+".cgenprobs.gz");
        wcl.write_category_mem_probs(model_fname+".cmemprobs.gz");
        write_class_unigram_counts(word_counts, wcl, model_fname+".ccounts.gz");

    }
    catch (string& e) {
        cerr << e << endl;
    }

    exit(0);
}
