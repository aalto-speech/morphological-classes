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
        void update(const CategoryProbs *probs);
        std::list<const CategoryProbs*> m_history;
        unsigned int m_history_length;
    };

    class HistoryToken {
    public:
        HistoryToken(const Ngram &ngram);
        double m_ll;
        int m_ngram_node;
    };

    std::vector<HistoryToken>
    propagate_history(const Ngram &ngram,
                      const CategoryHistory &history,
                      bool ngram_unk_states = true,
                      int num_tokens = 100,
                      double beam = FLT_MAX);

    double likelihood(const Ngram &ngram,
                      const Categories &wcs,
                      const std::vector<int> &intmap,
                      unsigned long int &num_words,
                      unsigned long int &num_oovs,
                      std::string word,
                      CategoryHistory &history,
                      bool ngram_unk_states = true,
                      int num_tokens = 100,
                      double beam = FLT_MAX);
}

#endif
