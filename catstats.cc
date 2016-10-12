#include <fstream>
#include <iostream>
#include <sstream>

#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Categories.hh"

using namespace std;


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: classseq [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT OUTPUT\n")
    ('p', "max-parses=INT", "arg", "10", "Maximum number of parses per sentence")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 5) config.print_help(stderr, 1);

    string ngramfname = config.arguments[0];
    string classpfname = config.arguments[1];
    string wordpfname = config.arguments[2];
    string infname = config.arguments[3];
    string outfname = config.arguments[4];

    int max_parses = config["max-parses"].get_int();

    WordClasses wcs;
    cerr << "Reading class probs.." << endl;
    wcs.read_class_probs(classpfname);
    cerr << "Reading word probs.." << endl;
    wcs.read_word_probs(wordpfname);

    cerr << "Reading class trigram model.." << endl;
//  Trigram tg;
//  tg.read_model(ngramfname);

    set<string> vocab; wcs.get_words(vocab, false);

    cerr << "Segmenting.." << endl;
    SimpleFileInput corpusf(infname);
    SimpleFileOutput outf(outfname);
    string line;
    while (corpusf.getline(line)) {
        vector<vector<string> > sent;
        stringstream ss(line);
        string word;
        vector<string> words = { "<s>", "<s>" };
        while (ss >> word) {
            if (word == "<s>" || word == "</s>") continue;
            words.push_back(word);
        }
        words.push_back("<s>");
        if ((int)words.size() > 100+3) continue;
        if ((int)words.size() == 3) continue;

        for (auto wit=words.begin(); wit != words.end(); ++wit) {
            if (*wit == "<s>") continue;
            if (vocab.find(*wit) == vocab.end())
                wit->assign("<unk>");
        }

        sent.push_back(words);

//        print_class_seqs(outf,
//                         sent, &tg, &wcs,
//                         100, 10.0, max_parses);
    }

    outf.close();

    exit(0);
}

