#include <boost/test/unit_test.hpp>

#include <iostream>
#include <vector>
#include <string>

#define private public
#include "ClassPerplexity.hh"
#undef private

using namespace std;


double DBL_ACCURACY = 0.0001;


// Class unigram, two word sentence without OOVs, one class per word
BOOST_AUTO_TEST_CASE(PerplexityTest1)
{
    cerr << endl;
    WordClasses wcs;
    wcs.read_class_probs("data/cprobs1.txt");
    wcs.read_word_probs("data/wprobs1.txt");

    Ngram ng;
    ng.read_arpa("data/classes.1g.wb.arpa.gz");

    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        indexmap[i] = ng.vocabulary_lookup[int2str(i)];

    SimpleFileInput sfi("data/test1.txt");
    string line;
    sfi.getline(line);
    int num_words, num_oovs;
    double ll = likelihood(line, ng, wcs, indexmap, num_words, num_oovs);

    BOOST_CHECK( num_words == 3 );
    BOOST_CHECK( num_oovs == 0 );
    // -7.952400 + -5.236393 + math.log(10.0) * (-1.073018 + -2.110189 + -1.072623) = -22.988203716316853
    BOOST_REQUIRE_CLOSE( -22.988203716316853, ll, DBL_ACCURACY );
}


// Class unigram, three word sentence with one OOV, one class per word
BOOST_AUTO_TEST_CASE(PerplexityTest2)
{
    cerr << endl;
    WordClasses wcs;
    wcs.read_class_probs("data/cprobs1.txt");
    wcs.read_word_probs("data/wprobs1.txt");

    Ngram ng;
    ng.read_arpa("data/classes.1g.wb.arpa.gz");

    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        indexmap[i] = ng.vocabulary_lookup[int2str(i)];

    SimpleFileInput sfi("data/test2.txt");
    string line;
    sfi.getline(line);
    int num_words, num_oovs;
    double ll = likelihood(line, ng, wcs, indexmap, num_words, num_oovs);

    BOOST_CHECK( num_words == 3 );
    BOOST_CHECK( num_oovs == 1 );
    // -7.952400 + -5.236393 + math.log(10.0) * (-1.073018 + -2.110189 + -1.072623) = -22.988203716316853
    BOOST_REQUIRE_CLOSE( -22.988203716316853, ll, DBL_ACCURACY );
}


// Class bigram, two word sentence without OOVs, one class per word
BOOST_AUTO_TEST_CASE(PerplexityTest3)
{
    cerr << endl;
    WordClasses wcs;
    wcs.read_class_probs("data/cprobs1.txt");
    wcs.read_word_probs("data/wprobs1.txt");

    Ngram ng;
    ng.read_arpa("data/classes.2g.wb.arpa.gz");

    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        indexmap[i] = ng.vocabulary_lookup[int2str(i)];

    SimpleFileInput sfi("data/test1.txt");
    string line;
    sfi.getline(line);
    int num_words, num_oovs;
    double ll = likelihood(line, ng, wcs, indexmap, num_words, num_oovs);

    BOOST_CHECK( num_words == 3 );
    BOOST_CHECK( num_oovs == 0 );
    // -7.952400 + -5.236393 + math.log(10.0) * (-1.122905 + -1.86108 + -1.140563) = -22.685915740138405
    BOOST_REQUIRE_CLOSE( -22.685915740138405, ll, DBL_ACCURACY );
}


// Class bigram, three word sentence with one OOV, one class per word
BOOST_AUTO_TEST_CASE(PerplexityTest4)
{
    cerr << endl;
    WordClasses wcs;
    wcs.read_class_probs("data/cprobs1.txt");
    wcs.read_word_probs("data/wprobs1.txt");

    Ngram ng;
    ng.read_arpa("data/classes.2g.wb.arpa.gz");

    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        indexmap[i] = ng.vocabulary_lookup[int2str(i)];

    SimpleFileInput sfi("data/test2.txt");
    string line;
    sfi.getline(line);
    int num_words, num_oovs;
    double ll = likelihood(line, ng, wcs, indexmap, num_words, num_oovs, true);

    BOOST_CHECK( num_words == 3 );
    BOOST_CHECK( num_oovs == 1 );
    // -7.952400 + -5.236393 + math.log(10.0) * (-1.122905 + -1.86108 + -0.8038033) = -21.91049787499726
    BOOST_REQUIRE_CLOSE( -21.91049787499726, ll, DBL_ACCURACY );
}


// Class bigram, three word sentence with one OOV, multiple classes per word
BOOST_AUTO_TEST_CASE(PerplexityTest5)
{
    cerr << endl;
    WordClasses wcs;
    wcs.read_class_probs("data/cprobs2.txt");
    wcs.read_word_probs("data/wprobs2.txt");

    Ngram ng;
    ng.read_arpa("data/classes.2g.wb.arpa.gz");

    vector<int> indexmap(wcs.num_classes());
    for (int i=0; i<(int)indexmap.size(); i++)
        indexmap[i] = ng.vocabulary_lookup[int2str(i)];

    SimpleFileInput sfi("data/test2.txt");
    string line;
    sfi.getline(line);
    int num_words, num_oovs;
    double ll = likelihood(line, ng, wcs, indexmap, num_words, num_oovs, true);

    BOOST_CHECK( num_words == 3 );
    BOOST_CHECK( num_oovs == 1 );
    // p(ulkona)
    // -7.952400 + math.log(10.0) * -1.122905 (+) -6.5 + math.log(10.0) * -4.410698
    // = -10.53798431384848 (+) -16.65600746449865
    // = -10.5357839302
    // p(sataa|ulkona) =
    //  -0.356675   + -5.236393 + math.log(10.0) * -1.86108
    // (+) -0.356675 + -5.5      + math.log(10.0) * -2.982406
    // (+) -1.203973 + -5.236393 + math.log(10.0) * -1.820686
    // (+) -1.203973 + -5.5      + math.log(10.0) * -3.181568
    // = -9.878363064869358 (+) -12.723918596856 (+) -10.632650442622957 (+) -14.029804049146879
    // = -9.44386397121
    // p(</s>|<unk>)
    // -0.8038033 * math.log(10.0)
    // -1.850825496279421
    // sum = -21.830473397689417
    BOOST_REQUIRE_CLOSE( -21.830473397, ll, DBL_ACCURACY );
}
