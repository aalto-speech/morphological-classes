#include <boost/test/unit_test.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <ctime>
#include <algorithm>

#define private public
#include "ExchangeAlgorithm.hh"
#undef private

using namespace std;


void
_assert_same(Exchange &e1,
             Exchange &e2)
{
    BOOST_CHECK_EQUAL(e1.m_num_classes, e2.m_num_classes);
    BOOST_CHECK(e1.m_vocabulary == e2.m_vocabulary);
    BOOST_CHECK(e1.m_vocabulary_lookup == e2.m_vocabulary_lookup);

    for (int i=0; i<(int)e1.m_classes.size(); i++)
        if (e1.m_classes[i].size() > 0)
            BOOST_CHECK( e1.m_classes[i] == e2.m_classes[i] );
    for (int i=0; i<(int)e2.m_classes.size(); i++)
        if (e2.m_classes[i].size() > 0)
            BOOST_CHECK( e1.m_classes[i] == e2.m_classes[i] );

    BOOST_CHECK( e1.m_word_classes == e2.m_word_classes );
    BOOST_CHECK( e1.m_word_counts == e2.m_word_counts );
    BOOST_CHECK( e1.m_word_bigram_counts == e2.m_word_bigram_counts );
    BOOST_CHECK( e1.m_word_rev_bigram_counts == e2.m_word_rev_bigram_counts );

    for (int i=0; i<(int)e1.m_class_counts.size(); i++)
        if (e1.m_class_counts[i] != 0)
            BOOST_CHECK_EQUAL(e1.m_class_counts[i], e2.m_class_counts[i]);
    for (int i=0; i<(int)e2.m_class_counts.size(); i++)
        if (e2.m_class_counts[i] != 0)
            BOOST_CHECK_EQUAL(e1.m_class_counts[i], e2.m_class_counts[i]);

    for (int i=0; i<(int)e1.m_class_bigram_counts.size(); i++)
        for (int j=0; j<(int)e1.m_class_bigram_counts[i].size(); j++)
            if (e1.m_class_bigram_counts[i][j] != 0)
                BOOST_CHECK_EQUAL(e1.m_class_bigram_counts[i][j], e2.m_class_bigram_counts[i][j]);
    for (int i=0; i<(int)e2.m_class_bigram_counts.size(); i++)
        for (int j=0; j<(int)e2.m_class_bigram_counts[i].size(); j++)
            if (e2.m_class_bigram_counts[i][j] != 0)
                BOOST_CHECK_EQUAL(e1.m_class_bigram_counts[i][j], e2.m_class_bigram_counts[i][j]);

    for (int i=0; i<(int)e1.m_class_word_counts.size(); i++)
        for (auto wit=e1.m_class_word_counts[i].begin(); wit != e1.m_class_word_counts[i].end(); ++wit) {
            BOOST_CHECK(wit->second != 0);
            BOOST_CHECK_EQUAL(wit->second, e2.m_class_word_counts[i][wit->first]);
        }
    for (int i=0; i<(int)e2.m_class_word_counts.size(); i++)
        for (auto cit=e2.m_class_word_counts[i].begin(); cit != e2.m_class_word_counts[i].end(); ++cit) {
            BOOST_CHECK(cit->second != 0);
            BOOST_CHECK_EQUAL(cit->second, e1.m_class_word_counts[i][cit->first]);
        }

    for (int i=0; i<(int)e1.m_word_class_counts.size(); i++)
        for (auto cit=e1.m_word_class_counts[i].begin(); cit != e1.m_word_class_counts[i].end(); ++cit) {
            BOOST_CHECK(cit->second != 0);
            BOOST_CHECK_EQUAL(cit->second, e2.m_word_class_counts[i][cit->first]);
        }
    for (int i=0; i<(int)e2.m_word_class_counts.size(); i++)
        for (auto cit=e2.m_word_class_counts[i].begin(); cit != e2.m_word_class_counts[i].end(); ++cit) {
            BOOST_CHECK(cit->second != 0);
            BOOST_CHECK_EQUAL(cit->second, e1.m_word_class_counts[i][cit->first]);
        }

    BOOST_CHECK_EQUAL( e1.log_likelihood(), e2.log_likelihood() );
}


// Test that merging classes works
BOOST_AUTO_TEST_CASE(DoMerge)
{
    cerr << endl;
    map<string, int> class_init = {{"a", 2}, {"b", 3}, {"c", 4}, {"d", 2}, {"e", 3}};
    Exchange e(3, class_init, "data/exchange1.txt");
    e.do_merge(3, 4);

    map<string, int> class_init_2 = {{"a", 2}, {"b", 3}, {"c", 3}, {"d", 2}, {"e", 3}};
    Exchange e2(2, class_init_2, "data/exchange1.txt");

    _assert_same(e, e2);
}


// Test evaluating likelihood change of a class merge
BOOST_AUTO_TEST_CASE(EvaluateMerge)
{
    cerr << endl;
    map<string, int> class_init = {{"a", 2}, {"b", 3}, {"c", 4}, {"d", 2}, {"e", 3}};
    Exchange e(3, class_init, "data/exchange1.txt");
    double initial_ll = e.log_likelihood();
    double hypo_ll_diff = e.evaluate_merge(3, 4);
    e.do_merge(3, 4);
    double merged_ll = e.log_likelihood();

    BOOST_REQUIRE_CLOSE(hypo_ll_diff, merged_ll-initial_ll, 0.001);
}


// Test that splitting classes works
BOOST_AUTO_TEST_CASE(DoSplit)
{
    map<string, int> class_init_2 = {{"a", 2}, {"b", 3}, {"c", 3}, {"d", 2}, {"e", 3}};
    Exchange e(2, class_init_2, "data/exchange1.txt");

    set<int> class1_words, class2_words;
    class1_words.insert(e.m_vocabulary_lookup["b"]);
    class1_words.insert(e.m_vocabulary_lookup["e"]);
    class2_words.insert(e.m_vocabulary_lookup["c"]);

    e.do_split(3, class1_words, class2_words);

    map<string, int> class_init = {{"a", 2}, {"b", 3}, {"c", 4}, {"d", 2}, {"e", 3}};
    Exchange e2(3, class_init, "data/exchange1.txt");

    _assert_same(e, e2);
}


// Test that splitting classes works
BOOST_AUTO_TEST_CASE(EvaluateSplit)
{
    map<string, int> class_init_2 = {{"a", 2}, {"b", 3}, {"c", 3}, {"d", 2}, {"e", 3}};
    Exchange e(2, class_init_2, "data/exchange1.txt");
    double initial_ll = e.log_likelihood();

    set<int> class1_words, class2_words;
    class1_words.insert(e.m_vocabulary_lookup["b"]);
    class1_words.insert(e.m_vocabulary_lookup["e"]);
    class2_words.insert(e.m_vocabulary_lookup["c"]);
    double hypo_ll_diff = e.evaluate_split(3, class1_words, class2_words);

    map<string, int> class_init = {{"a", 2}, {"b", 3}, {"c", 4}, {"d", 2}, {"e", 3}};
    Exchange e2(3, class_init, "data/exchange1.txt");
    double splitted_ll = e2.log_likelihood();

    BOOST_REQUIRE_CLOSE(hypo_ll_diff, splitted_ll-initial_ll, 0.001);
}

