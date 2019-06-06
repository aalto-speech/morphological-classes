#ifndef PROJECT_DEFS
#define PROJECT_DEFS

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <sstream>
#include <vector>

#include "io.hh"
#include "Ngram.hh"

typedef float flt_type;

static flt_type MIN_LOG_PROB = -1000.0;
static flt_type LP_PRUNE_LIMIT = -50.0;

#define SENTENCE_BEGIN_SYMBOL "<s>"
#define SENTENCE_END_SYMBOL "</s>"
#define UNK_SYMBOL "<unk>"
#define CAP_UNK_SYMBOL "<UNK>"

// Return log(X+Y) where a=log(X) b=log(Y)
static flt_type add_log_domain_probs(flt_type a, flt_type b)
{
    flt_type delta = a-b;
    if (delta>0) {
        b = a;
        delta = -delta;
    }
    return b+log1p(exp(delta));
}

// Return log(X-Y) where a=log(X) b=log(Y)
static flt_type sub_log_domain_probs(flt_type a, flt_type b)
{
    flt_type delta = b-a;
    if (delta>0) {
        fprintf(stderr, "invalid call to sub_log_domain_probs, a should be bigger than b (a=%f,b=%f)\n", a, b);
        exit(1);
    }
    return a+log1p(-exp(delta));
}

static int str2int(std::string str)
{
    int val;
    std::istringstream numstr(str);
    numstr >> val;
    return val;
}

static std::string int2str(int a)
{
    std::ostringstream temp;
    temp << a;
    return temp.str();
}

static int
read_class_memberships(
        std::string fname,
        std::map<std::string, std::pair<int, flt_type>>& class_memberships)
{
    SimpleFileInput wcf(fname);

    std::string line;
    int max_class = 0;
    while (wcf.getline(line)) {
        std::stringstream ss(line);
        std::string word;
        int clss;
        flt_type prob;
        ss >> word >> clss >> prob;
        class_memberships[word] = std::make_pair(clss, prob);
        max_class = std::max(max_class, clss);
    }
    return max_class+1;
}

// The class indexes are stored as strings in the n-gram class
static std::vector<int>
get_class_index_map(
        int num_classes,
        Ngram& cngram)
{
    std::vector<int> indexmap(num_classes);
    for (int i = 0; i<(int) indexmap.size(); i++)
        if (cngram.vocabulary_lookup.find(int2str(i))!=cngram.vocabulary_lookup.end())
            indexmap[i] = cngram.vocabulary_lookup[int2str(i)];
        else
            std::cerr << "warning, class not found in the n-gram: " << i << std::endl;
    return indexmap;
}

#endif /* PROJECT_DEFS */
