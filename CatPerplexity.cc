#include <cfloat>
#include <numeric>

#include "CatPerplexity.hh"

using namespace std;


namespace CatPerplexity {

    CategoryHistory::CategoryHistory(const Ngram &ngram) {
        m_history_length = ngram.max_order - 1;
    }

    void
    CategoryHistory::update(const CategoryProbs *probs) {
        m_history.push_back(probs);
        while (m_history.size() > m_history_length) {
            m_history.pop_front();
        }
    }

    HistoryToken::HistoryToken(const Ngram &ngram) {
        m_ll = 0.0;
        m_ngram_node = ngram.sentence_start_node;
    }

    vector<HistoryToken>
    propagate_history(const Ngram &ngram,
                      const CategoryHistory &history,
                      bool ngram_unk_states,
                      int num_tokens,
                      double beam)
    {
        vector<HistoryToken> final_tokens;
        return final_tokens;
    }

    double
    likelihood(const Ngram &ngram,
               const Categories &wcs,
               const vector<int> &intmap,
               unsigned long int &num_words,
               unsigned long int &num_oovs,
               string word,
               CategoryHistory &history,
               bool ngram_unk_states,
               int num_tokens,
               double beam)
    {

        bool sentence_end = false;
        bool unk = false;
        auto cgenit = wcs.m_category_gen_probs.find(word);
        auto cmemit = wcs.m_category_mem_probs.find(word);

        if (word == "</s>")
            sentence_end = true;
        else if (word == "<unk>" || word == "<UNK>")
            unk = true;
        else if (cgenit == wcs.m_category_gen_probs.end() || cgenit->second.size() == 0)
            unk = true;
        else if (cmemit == wcs.m_category_mem_probs.end() || cmemit->second.size() == 0)
            unk = true;

        vector<CatPerplexity::HistoryToken> tokens
                = propagate_history(ngram,
                                    history,
                                    ngram_unk_states,
                                    num_tokens,
                                    beam);

        if (unk) {
            num_oovs++;
            history.update(nullptr);
            return 0.0;
        }
        else if (sentence_end) {
            double total_ll = -FLT_MAX;
            for (auto tit = tokens.begin(); tit != tokens.end(); ++tit) {
                double ll = tit->m_ll;
                double ngram_ll = 0.0;
                ngram.score(tit->m_ngram_node, ngram.sentence_end_symbol_idx, ngram_ll);
                ll += ngram_ll * log(10.0);
                total_ll = add_log_domain_probs(total_ll, ll);
            }
            return total_ll;
        }
        else {
            double total_ll = -FLT_MAX;
            for (auto tit = tokens.begin(); tit != tokens.end(); ++tit) {
                for (auto cit = cmemit->second.begin(); cit != cmemit->second.end(); ++cit) {
                    double ll = tit->m_ll;
                    double ngram_ll = 0.0;
                    ngram.score(tit->m_ngram_node, intmap[cit->first], ngram_ll);
                    ll += ngram_ll * log(10.0);
                    ll += cit->second;
                    total_ll = add_log_domain_probs(total_ll, ll);
                }
            }
            num_words++;
            history.update(&(cgenit->second));
            return total_ll;
        }
    }
}
