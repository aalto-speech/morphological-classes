#ifndef CLASS_PPL
#define CLASS_PPL

#include <vector>
#include <string>

#include "Ngram.hh"
#include "Categories.hh"


void score(const std::vector<WordClassProbs*> &probs,
           const Ngram &ngram,
           const std::vector<int> &indexmap,
           int ngram_node,
           bool sentence_end,
           flt_type &total_score,
           bool ngram_unk_states=false,
           flt_type beam=100.0);

flt_type likelihood(std::string &sent,
                    Ngram &ngram,
                    WordClasses &wcs,
                    std::vector<int> &intmap,
                    int &num_words,
                    int &num_oovs,
                    bool ngram_unk_states=false);

flt_type likelihood(std::string &sent,
                    Ngram &ngram,
                    WordClasses &wcs,
                    std::vector<int> &intmap,
                    std::vector<flt_type> &word_lls,
                    int &num_words,
                    int &num_oovs,
                    bool ngram_unk_states=false);

#endif
