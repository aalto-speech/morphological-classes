#include <boost/test/unit_test.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <ctime>
#include <algorithm>

#define private public
#include "Splitting.hh"
#undef private

using namespace std;

void
_assert_same(Splitting& s1,
        Splitting& s2)
{
    BOOST_CHECK_EQUAL(s1.m_num_classes, s2.m_num_classes);
    BOOST_CHECK(s1.m_vocabulary==s2.m_vocabulary);
    BOOST_CHECK(s1.m_vocabulary_lookup==s2.m_vocabulary_lookup);

    for (int i = 0; i<(int) s1.m_classes.size(); i++)
        if (s1.m_classes[i].size()>0)
            BOOST_CHECK(s1.m_classes[i]==s2.m_classes[i]);
    for (int i = 0; i<(int) s2.m_classes.size(); i++)
        if (s2.m_classes[i].size()>0)
            BOOST_CHECK(s1.m_classes[i]==s2.m_classes[i]);

    BOOST_CHECK(s1.m_word_classes==s2.m_word_classes);
    BOOST_CHECK(s1.m_word_counts==s2.m_word_counts);
    BOOST_CHECK(s1.m_word_bigram_counts==s2.m_word_bigram_counts);
    BOOST_CHECK(s1.m_word_rev_bigram_counts==s2.m_word_rev_bigram_counts);

    for (int i = 0; i<(int) s1.m_class_counts.size(); i++)
        if (s1.m_class_counts[i]!=0)
            BOOST_CHECK_EQUAL(s1.m_class_counts[i], s2.m_class_counts[i]);
    for (int i = 0; i<(int) s2.m_class_counts.size(); i++)
        if (s2.m_class_counts[i]!=0)
            BOOST_CHECK_EQUAL(s1.m_class_counts[i], s2.m_class_counts[i]);

    for (int i = 0; i<(int) s1.m_class_bigram_counts.size(); i++)
        for (int j = 0; j<(int) s1.m_class_bigram_counts[i].size(); j++)
            if (s1.m_class_bigram_counts[i][j]!=0)
                BOOST_CHECK_EQUAL(s1.m_class_bigram_counts[i][j], s2.m_class_bigram_counts[i][j]);
    for (int i = 0; i<(int) s2.m_class_bigram_counts.size(); i++)
        for (int j = 0; j<(int) s2.m_class_bigram_counts[i].size(); j++)
            if (s2.m_class_bigram_counts[i][j]!=0)
                BOOST_CHECK_EQUAL(s1.m_class_bigram_counts[i][j], s2.m_class_bigram_counts[i][j]);

    for (int i = 0; i<(int) s1.m_class_word_counts.size(); i++)
        for (auto wit = s1.m_class_word_counts[i].begin(); wit!=s1.m_class_word_counts[i].end(); ++wit) {
            BOOST_CHECK(wit->second!=0);
            BOOST_CHECK_EQUAL(wit->second, s2.m_class_word_counts[i][wit->first]);
        }
    for (int i = 0; i<(int) s2.m_class_word_counts.size(); i++)
        for (auto cit = s2.m_class_word_counts[i].begin(); cit!=s2.m_class_word_counts[i].end(); ++cit) {
            BOOST_CHECK(cit->second!=0);
            BOOST_CHECK_EQUAL(cit->second, s1.m_class_word_counts[i][cit->first]);
        }

    for (int i = 0; i<(int) s1.m_word_class_counts.size(); i++)
        for (auto cit = s1.m_word_class_counts[i].begin(); cit!=s1.m_word_class_counts[i].end(); ++cit) {
            BOOST_CHECK(cit->second!=0);
            BOOST_CHECK_EQUAL(cit->second, s2.m_word_class_counts[i][cit->first]);
        }
    for (int i = 0; i<(int) s2.m_word_class_counts.size(); i++)
        for (auto cit = s2.m_word_class_counts[i].begin(); cit!=s2.m_word_class_counts[i].end(); ++cit) {
            BOOST_CHECK(cit->second!=0);
            BOOST_CHECK_EQUAL(cit->second, s1.m_word_class_counts[i][cit->first]);
        }

    BOOST_CHECK_EQUAL(s1.log_likelihood(), s2.log_likelihood());
}

// Test that splitting classes works
BOOST_AUTO_TEST_CASE(DoSplit)
        {
                map<string, int>class_init_2 = {{"a", 2}, {"b", 3}, {"c", 3}, {"d", 2}, {"e", 3}};
        Splitting splitting(2, class_init_2, "data/exchange1.txt");

        set<int> class1_words, class2_words;
        class1_words.insert(splitting.m_vocabulary_lookup["b"]);
        class1_words.insert(splitting.m_vocabulary_lookup["e"]);
        class2_words.insert(splitting.m_vocabulary_lookup["c"]);

        splitting.do_split(3, class1_words, class2_words);

        map<string, int> class_init = {{ "a", 2 }, { "b", 3 }, { "c", 4 }, { "d", 2 }, { "e", 3 }};
        Splitting splitting2(3, class_init, "data/exchange1.txt");

        _assert_same(splitting, splitting2);
        }

