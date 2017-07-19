#ifndef CATEGORY_PPL
#define CATEGORY_PPL

#include <cfloat>
#include <string>
#include <vector>

#include "Ngram.hh"
#include "Categories.hh"


namespace CatPerplexity {

    struct Token {
        double m_acc_ll;
        int m_ngram_node;
        std::vector<double> m_cat_gen_lls;
    };

    double likelihood(const Ngram &ngram,
                      const Categories &wcs,
                      const std::vector<int> &intmap,
                      long long unsigned int &num_words,
                      long long unsigned int &num_oovs,
                      std::string word,
                      std::vector <CatPerplexity::Token> &tokens,
                      bool ngram_unk_states = true,
                      double beam = FLT_MAX);
}

#endif
