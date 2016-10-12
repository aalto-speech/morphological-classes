#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

#define private public
#include "Classes.hh"
#undef private

using namespace std;


struct AddValues
{
  template<class Value, class Pair>
  Value operator()(Value value, const Pair& pair) const
  {
    return value + pair.second;
  }
};


void set_wcs_fixture(WordClasses &wcs)
{
    wcs.m_num_classes = 5;

    map<string, pair<int, map<int, double> > > words;
    map<int,double> sent_class_gen_probs = { {START_CLASS,1.0} };
    words["<s>"] = make_pair(10, sent_class_gen_probs);
    map<int,double> unk_class_gen_probs = { {UNK_CLASS,1.0} };
    words["<unk>"] = make_pair(2, unk_class_gen_probs);
    map<int,double> a_class_gen_probs = { {2,1.0} };
    words["a"] = make_pair(5, a_class_gen_probs);
    map<int,double> b_class_gen_probs = { {3,1.0} };
    words["b"] = make_pair(4, b_class_gen_probs);
    map<int,double> c_class_gen_probs = { {2,0.5}, {3,0.5} };
    words["c"] = make_pair(4, c_class_gen_probs);
    map<int,double> d_class_gen_probs = { {2,0.8}, {3,0.2} };
    words["d"] = make_pair(3, d_class_gen_probs);
    map<int,double> e_class_gen_probs = { {3,0.5}, {4,0.5} };
    words["e"] = make_pair(3, e_class_gen_probs);
    map<int,double> f_class_gen_probs = { {3,0.2}, {4,0.8} };
    words["f"] = make_pair(2, f_class_gen_probs);

    map<int, double> class_totals;
    for (auto wit=words.begin(); wit != words.end(); ++wit) {
        for (auto cit=wit->second.second.begin(); cit != wit->second.second.end(); ++cit) {
            wcs.m_class_gen_probs[wit->first][cit->first] = log(cit->second);
            class_totals[cit->first] += wit->second.first * cit->second;
        }
    }

    for (auto wit=words.begin(); wit != words.end(); ++wit) {
        for (auto cit=wit->second.second.begin(); cit != wit->second.second.end(); ++cit) {
            double freq = wit->second.first * cit->second;
            wcs.m_class_memberships[wit->first][cit->first] =
                    log(freq) - log(class_totals[cit->first]);
        }
    }
}


// Category model corpus reading
BOOST_AUTO_TEST_CASE(CategoryTest1)
{
    WordClasses wcs;
    set_wcs_fixture(wcs);
    wcs.assert_class_probs();
    wcs.assert_word_probs();

    vector<vector<string> > sents;
    int num_unk_tokens, num_unk_types, num_word_tokens, num_word_types;
    set<string> vocab;
    wcs.get_words(vocab, false);

    read_sents("data/merge_c1.txt", sents, 100,
               &vocab,
               &num_word_tokens, &num_word_types,
               &num_unk_tokens, &num_unk_types);

    BOOST_CHECK_EQUAL( 8, vocab.size() );
    BOOST_CHECK_EQUAL( 60, num_word_tokens );
    BOOST_CHECK_EQUAL( 6, num_word_types );
    BOOST_CHECK_EQUAL( 0, num_unk_tokens );
    BOOST_CHECK_EQUAL( 0, num_unk_types );
}


// Category model first n-gram estimation
BOOST_AUTO_TEST_CASE(CategoryTest2)
{
    WordClasses wcs;
    set_wcs_fixture(wcs);
    wcs.assert_class_probs();
    wcs.assert_word_probs();

    vector<vector<string> > sents;
    set<string> vocab;
    wcs.get_words(vocab, false);

    read_sents("data/merge_c1.txt", sents, 100, &vocab);

    WordClasses word_stats(wcs.m_num_classes);
    Unigram ngram_stats;

    flt_type ll = collect_stats_thr(sents, new Unigram(wcs.m_num_classes), &wcs,
                                    &ngram_stats, &word_stats,
                                    1, 100, 100, 100, false);

    map<string, WordClassProbs> &stats = word_stats.m_stats;
    BOOST_CHECK_EQUAL( 7, stats.size() );
    BOOST_REQUIRE_CLOSE( 10.0, accumulate(stats["a"].begin(), stats["a"].end(), 0.0, AddValues()), 0.001 );
    BOOST_REQUIRE_CLOSE( 4.0, accumulate(stats["b"].begin(), stats["b"].end(), 0.0, AddValues()), 0.001 );
    BOOST_REQUIRE_CLOSE( 15.0, accumulate(stats["c"].begin(), stats["c"].end(), 0.0, AddValues()), 0.001 );
    BOOST_REQUIRE_CLOSE( 9.0, accumulate(stats["d"].begin(), stats["d"].end(), 0.0, AddValues()), 0.001 );
    BOOST_REQUIRE_CLOSE( 11.0, accumulate(stats["e"].begin(), stats["e"].end(), 0.0, AddValues()), 0.001 );
    BOOST_REQUIRE_CLOSE( 11.0, accumulate(stats["f"].begin(), stats["f"].end(), 0.0, AddValues()), 0.001 );
    BOOST_REQUIRE_CLOSE( 20.0, accumulate(stats["<s>"].begin(), stats["<s>"].end(), 0.0, AddValues()), 0.001 );

    BOOST_CHECK_EQUAL( 4, ngram_stats.m_unigrams.size() );
    double total = ngram_stats.m_unigrams[2];
    total += ngram_stats.m_unigrams[3];
    total += ngram_stats.m_unigrams[4];
    BOOST_CHECK( ngram_stats.m_unigrams[2] > 10.0 );
    BOOST_CHECK( ngram_stats.m_unigrams[3] > 10.0 );
    BOOST_CHECK( ngram_stats.m_unigrams[4] > 10.0 );
    BOOST_REQUIRE_CLOSE( 60.0, total, 0.001);
    BOOST_REQUIRE_CLOSE( ngram_stats.m_unigrams[START_CLASS], 10.0, 0.001 );
    //BOOST_REQUIRE_CLOSE( ngram_stats.m_unigrams[UNK_CLASS], 0.0, 0.001 );
}

