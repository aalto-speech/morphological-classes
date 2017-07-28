#include <cfloat>
#include <numeric>
#include <queue>

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

    bool operator<(const HistoryToken& lhs, const HistoryToken& rhs)
    {
        return lhs.m_ll < rhs.m_ll;
    }

    vector<HistoryToken>
    propagate_history(const Ngram &ngram,
                      const CategoryHistory &history,
                      const vector<int> &intmap,
                      bool root_unk_states,
                      int num_tokens,
                      double beam)
    {
        priority_queue<HistoryToken> init_tokens;
        init_tokens.push(HistoryToken(ngram));

        for (auto hit = history.m_history.begin(); hit != history.m_history.end(); ++hit) {
            priority_queue<HistoryToken> propagated_tokens;
            const CategoryProbs *cats = *hit;
            int tcount = 0;
            if (cats == nullptr) {
                while (init_tokens.size() > 0  && tcount++ < num_tokens) {
                    HistoryToken tok = init_tokens.top();
                    init_tokens.pop();
                    if (root_unk_states)
                        tok.m_ngram_node = ngram.root_node;
                    else
                        tok.m_ngram_node = ngram.advance(tok.m_ngram_node, ngram.unk_symbol_idx);
                    propagated_tokens.push(tok);
                }
            }
            else {
                while (init_tokens.size() > 0 && tcount++ < num_tokens) {
                    HistoryToken tok = init_tokens.top();
                    init_tokens.pop();
                    for (auto cit = cats->begin(); cit != cats->end(); ++cit) {
                        HistoryToken ctok = tok;
                        ctok.m_ngram_node = ngram.advance(ctok.m_ngram_node, intmap[cit->first]);
                        ctok.m_ll += cit->second;
                        propagated_tokens.push(ctok);
                    }
                }
            }

            init_tokens = propagated_tokens;
        }

        int ftcount = 0;
        vector<HistoryToken> final_tokens;
        while (init_tokens.size() > 0 && ftcount++ < num_tokens) {
            final_tokens.push_back(init_tokens.top());
            init_tokens.pop();
        }
        return final_tokens;
    }

    double
    likelihood(const LNNgram &ngram,
               const Categories &wcs,
               const vector<int> &intmap,
               unsigned long int &num_words,
               unsigned long int &num_oovs,
               string word,
               CategoryHistory &history,
               bool root_unk_states,
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
                                    intmap,
                                    root_unk_states,
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
                ngram.score(tit->m_ngram_node, ngram.sentence_end_symbol_idx, ll);
                total_ll = add_log_domain_probs(total_ll, ll);
            }
            return total_ll;
        }
        else {
            double total_ll = -FLT_MAX;
            for (auto tit = tokens.begin(); tit != tokens.end(); ++tit) {
                for (auto cit = cmemit->second.begin(); cit != cmemit->second.end(); ++cit) {
                    double ll = tit->m_ll;
                    ngram.score(tit->m_ngram_node, intmap[cit->first], ll);
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
