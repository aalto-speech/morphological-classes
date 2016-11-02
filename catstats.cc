#include <fstream>
#include <iostream>
#include <sstream>

#include "defs.hh"
#include "io.hh"
#include "conf.hh"
#include "Categories.hh"
#include "Ngram.hh"

using namespace std;


int main(int argc, char* argv[]) {

    conf::Config config;
    config("usage: catstats [OPTION...] CAT_ARPA CGENPROBS CMEMPROBS INPUT OUTPUT\n")
    ('p', "max-parses=INT", "arg", "10", "Maximum number of parses per sentence")
    ('h', "help", "", "", "display help");
    config.default_parse(argc, argv);
    if (config.arguments.size() != 5) config.print_help(stderr, 1);

    string cngramfname = config.arguments[0];
    string cgenpfname = config.arguments[1];
    string cmempfname = config.arguments[2];
    string infname = config.arguments[3];
    string outfname = config.arguments[4];

    int max_parses = config["max-parses"].get_int();

    Categories wcs;
    cerr << "Reading class generation probs.." << endl;
    wcs.read_class_gen_probs(cgenpfname);
    cerr << "Reading class membership probs.." << endl;
    wcs.read_class_mem_probs(cmempfname);

    cerr << "Reading class n-gram model.." << endl;
    Ngram cngram;
    cngram.read_arpa(cngramfname);

    set<string> vocab; wcs.get_words(vocab, false);

    cerr << "Segmenting.." << endl;
    SimpleFileInput corpusf(infname);
    SimpleFileOutput outf(outfname);
    string line;
    while (corpusf.getline(line)) {
        vector<vector<string> > sent;
        stringstream ss(line);
        string word;
        vector<string> words;
        while (ss >> word) {
            if (word == "<s>" || word == "</s>") continue;
            words.push_back(word);
        }
        words.push_back("</s>");
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

