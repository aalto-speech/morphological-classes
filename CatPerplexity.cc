#include <cfloat>

#include "CatPerplexity.hh"

using namespace std;


void score(const vector<CategoryProbs*> &probs,
           const Ngram &ngram,
           const vector<int> &indexmap,
           int ngram_node,
           bool sentence_end,
           flt_type &total_score,
           bool ngram_unk_states,
           flt_type beam)
{
    struct CToken {
        flt_type m_prob;
        int m_ngram_node;
    };

    vector<vector<CToken> > tokens(probs.size());
    CToken init_token;
    init_token.m_prob = 0.0;
    init_token.m_ngram_node = ngram_node;
    tokens[0].push_back(init_token);

    flt_type prev_best = 0.0;
    flt_type curr_best = -FLT_MAX;
    for (int i=0; i<(int)probs.size()-1; i++) {
        vector<CToken> &curr_tokens = tokens[i];
        vector<CToken> &target_tokens = tokens[i+1];
        for (auto tit=curr_tokens.begin(); tit != curr_tokens.end(); tit++) {
            if (tit->m_prob + beam < prev_best) continue;
            CategoryProbs& pos_probs = *(probs[i]);
            for (auto pit = pos_probs.begin(); pit != pos_probs.end(); ++pit) {
                CToken tok;
                tok.m_prob = tit->m_prob + pit->second;
                //if (pit->first != UNK_CLASS || ngram_unk_states)
                //    tok.m_ngram_node = ngram.advance(tit->m_ngram_node, indexmap[pit->first]);
                //else tok.m_ngram_node = ngram.root_node;
                target_tokens.push_back(tok);
                curr_best = max(curr_best, tok.m_prob);
            }
        }
        prev_best = curr_best;
        curr_best = -FLT_MAX;
    }

    vector<CToken> &curr_tokens = tokens.back();
    for (auto tit=curr_tokens.begin(); tit != curr_tokens.end(); tit++) {
        if (tit->m_prob + beam < prev_best) continue;
        CategoryProbs& pos_probs = *(probs.back());
        for (auto pit = pos_probs.begin(); pit != pos_probs.end(); ++pit) {
            int csym;
            if (sentence_end) csym = ngram.vocabulary_lookup.at("</s>");
            else csym = indexmap[pit->first];
            flt_type ngram_score = 0.0;
            ngram.score(tit->m_ngram_node, csym, ngram_score);
            ngram_score *= log(10.0);
            total_score = add_log_domain_probs(total_score, ngram_score + tit->m_prob + pit->second);
        }
    }
}


flt_type
likelihood(string &sent,
           Ngram &ngram,
           Categories &wcs,
           vector<int> &indexmap,
           std::vector<flt_type> &word_lls,
           int &num_words,
           int &num_oovs,
           bool ngram_unk_states)
{
    word_lls.clear();
    num_words = 0;
    num_oovs = 0;
    int order = ngram.order();

    int start_node = ngram.advance(ngram.root_node, ngram.vocabulary_lookup["<s>"]);

    stringstream ss(sent);
    vector<string> words;
    string word;
    while (ss >> word) {
        if (word == "<s>") continue;
        if (word == "</s>") continue;
        if (wcs.m_class_gen_probs.find(word) == wcs.m_class_gen_probs.end()
            || word == "<unk>"  || word == "<UNK>")
        {
            words.push_back("<unk>");
            num_oovs++;
        }
        else {
            words.push_back(word);
            num_words++;
        }
    }
    num_words++;

    flt_type sent_ll = 0.0;

    CategoryProbs *special_prob = new CategoryProbs;
    (*special_prob)[-1] = 0.0;

    for (int i=0; i<(int)words.size(); i++) {
        if (words[i] == "<unk>") continue;
        std::vector<CategoryProbs*> probs;
        // FIXME
        for (int ctxi = max(0, i-order+1); ctxi<i; ctxi++)
            if (words[ctxi] == "<unk>") probs.push_back(special_prob);
            else probs.push_back(&(wcs.m_class_gen_probs.at(words[ctxi])));
        probs.push_back(&(wcs.m_class_mem_probs.at(words[i])));
        flt_type word_ll = -1000;
        score(probs, ngram, indexmap, start_node, false, word_ll, ngram_unk_states);
        sent_ll += word_ll;
        word_lls.push_back(word_ll);
    }

    std::vector<CategoryProbs*> probs;
    // FIXME
    for (int ctxi = max(0, (int)words.size()-order+1); ctxi<(int)words.size(); ctxi++)
        if (words[ctxi] == "<unk>") probs.push_back(special_prob);
        else probs.push_back(&(wcs.m_class_gen_probs.at(words[ctxi])));
    // FIXME
    probs.push_back(special_prob);
    flt_type word_ll = -1000;
    score(probs, ngram, indexmap, start_node, true, word_ll, ngram_unk_states);
    sent_ll += word_ll;
    word_lls.push_back(word_ll);

    return sent_ll;
}


flt_type
likelihood(string &sent,
           Ngram &ngram,
           Categories &wcs,
           vector<int> &indexmap,
           int &num_words,
           int &num_oovs,
           bool ngram_unk_states)
{
    vector<flt_type> word_lls;
    return likelihood(sent, ngram, wcs, indexmap, word_lls,
                      num_words, num_oovs, ngram_unk_states);
}

