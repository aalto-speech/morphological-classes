#include <cfloat>
#include <numeric>

#include "CatPerplexity.hh"

using namespace std;


namespace CatPerplexity {

    double
    likelihood(const Ngram &ngram,
               const Categories &wcs,
               const vector<int> &intmap,
               unsigned long int &num_words,
               unsigned long int &num_oovs,
               string word,
               vector<CatPerplexity::Token> &tokens,
               bool ngram_unk_states,
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

        vector<CatPerplexity::Token> propagated_tokens;
        double best_acc_ll = -FLT_MAX;
        double total_ll = -1000.0;
        if (unk) {
            for (auto tit = tokens.begin(); tit != tokens.end(); ++tit) {
                CatPerplexity::Token tok = *tit;
                if (ngram_unk_states)
                    tok.m_ngram_node = ngram.advance(tok.m_ngram_node, ngram.unk_symbol_idx);
                else
                    tok.m_ngram_node = ngram.root_node;
                tok.m_cat_gen_lls.push_back(0.0);
                while ((int)tok.m_cat_gen_lls.size() > (ngram.max_order - 1))
                    tok.m_cat_gen_lls.pop_front();
                propagated_tokens.push_back(tok);
            }
            num_oovs++;
            total_ll = 0.0;
        }
        else if (sentence_end) {
            for (auto tit = tokens.begin(); tit != tokens.end(); ++tit) {
                CatPerplexity::Token tok = *tit;
                double ll = 0.0;
                for (auto cgit = tit->m_cat_gen_lls.begin(); cgit != tit->m_cat_gen_lls.end(); ++cgit)
                    ll += *cgit;
                double ngram_ll = 0.0;
                tok.m_ngram_node = ngram.score(tok.m_ngram_node,
                    ngram.sentence_end_symbol_idx, ngram_ll);
                ll += ngram_ll * log(10.0);
                tok.m_cat_gen_lls.push_back(0.0);
                while ((int)tok.m_cat_gen_lls.size() > (ngram.max_order - 1))
                    tok.m_cat_gen_lls.pop_front();
                tok.m_acc_ll += ll;
                total_ll = add_log_domain_probs(total_ll, ll);
                best_acc_ll = std::max(best_acc_ll, tok.m_acc_ll);
                propagated_tokens.push_back(tok);
            }
        }
        else {
            for (auto tit = tokens.begin(); tit != tokens.end(); ++tit) {
                double gen_ll = 0.0;
                for (auto cgit = tit->m_cat_gen_lls.begin(); cgit != tit->m_cat_gen_lls.end(); ++cgit)
                    gen_ll += *cgit;
                for (auto cit = cmemit->second.begin(); cit != cmemit->second.end(); ++cit) {
                    CatPerplexity::Token tok = *tit;
                    double ll = gen_ll;
                    double ngram_ll = 0.0;
                    tok.m_ngram_node = ngram.score(tok.m_ngram_node,
                        intmap[cit->first], ngram_ll);
                    ll += ngram_ll * log(10.0);
                    ll += cit->second;
                    tok.m_cat_gen_lls.push_back(cgenit->second.at(cit->first));
                    while ((int)tok.m_cat_gen_lls.size() > (ngram.max_order - 1))
                        tok.m_cat_gen_lls.pop_front();
                    tok.m_acc_ll += ll;
                    total_ll = add_log_domain_probs(total_ll, ll);
                    best_acc_ll = std::max(best_acc_ll, tok.m_acc_ll);
                    propagated_tokens.push_back(tok);
                }
            }
            num_words++;
        }

        tokens.clear();
        for (auto ptit = propagated_tokens.begin(); ptit != propagated_tokens.end(); ++ptit)
            if (ptit->m_acc_ll > (best_acc_ll - beam)) tokens.push_back(*ptit);

        return total_ll;
    }
}
