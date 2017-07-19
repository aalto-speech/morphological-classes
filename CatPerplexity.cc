#include <cfloat>

#include "CatPerplexity.hh"

using namespace std;


namespace CatPerplexity {

    double
    likelihood(const Ngram &ngram,
               const Categories &wcs,
               const std::vector<int> &intmap,
               long long unsigned int &num_words,
               long long unsigned int &num_oovs,
               std::string word,
               std::vector <CatPerplexity::Token> &tokens,
               bool ngram_unk_states,
               double beam)
    {
        return 0.0;
    }
}
