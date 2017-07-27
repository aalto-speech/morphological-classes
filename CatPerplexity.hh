#ifndef CATEGORY_PPL
#define CATEGORY_PPL

#include <cfloat>
#include <list>
#include <string>
#include <vector>

#include "Ngram.hh"
#include "Categories.hh"


namespace CatPerplexity {

    class CategoryHistory {
    public:
        CategoryHistory(const Ngram &ngram);
        void update(CategoryProbs *probs);
        std::list<CategoryProbs*> m_history;
        unsigned int m_history_length;
    };

    class Token {
    public:
        Token(const Ngram &ngram)
            : m_acc_ll(0.0),
              m_ngram_node(ngram.sentence_start_node) { };
        double m_acc_ll;
        int m_ngram_node;
        std::list<double> m_cat_gen_lls;
    };

    double likelihood(const Ngram &ngram,
                      const Categories &wcs,
                      const std::vector<int> &intmap,
                      unsigned long int &num_words,
                      unsigned long int &num_oovs,
                      std::string word,
                      std::vector <CatPerplexity::Token> &tokens,
                      bool ngram_unk_states = true,
                      double beam = FLT_MAX);
}

#endif
