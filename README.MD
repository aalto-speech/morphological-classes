# Morphological word class training

This project contains software for training word classes from a text corpus.
For instance morphologically motivated classes can be learned using an expectation-maximization training scheme if the classes are initialized using a morphological analyzer.
The model formulation allows multiple classes per word during the EM training and the order of the n-gram statistics is not limited.
The default settings use trigram statistics over the class sequences in the final phases of the training.
For further refining the classes, the model can be constrained to allow only one class per word.
In that case class merging and splitting operations and also exchange algorithm can be run using bigram statistics.

### Compilation

The project can be compiled using standard GNU compilation toolchain, i.e. `g++` and `make`.  
In Windows, the MinGW package may be used.
For compiling the executables and unit tests, copy Makefile.local.example to Makefile.local and run make  
`cp Makefile.local.example Makefile.local`  
`make`  
The zlib library and headers are required. On linux systems, these are often included in packages `zlib1g` and `zlib1g-dev`.  
In addition, for running the unit tests, Boost unit testing libraries must be installed.  
On linux systems, these are often included in packages `libboost-test` and `libboost-test-dev`.
The compilation may be tuned by modifying the Makefile.local file.  
If you do not need to run the unit tests, set the environment variable NO_UNIT_TESTS to some value.

### Model formulation for the expectation-maximization training

The form of the model is:  
`P(w| w-2 .. w-1) = sum_j P(w|c_j) P( c_j| w-2 .. w-1)`,  
where j are the possible classes for the word `w`  
`P(c_j|w-2 .. w-1) = sum_s P( c_j | s) * P(s| w-2 .. w-1)`,  
`s` denoting possible class sequences generated by the word sequence `w-2 .. w-1`  
the class sequence generation probabilities are assumed to be dependent only on the word in the exact position
`P(s|w-2 .. w-1) = prod P(k) (c_k | w_k)`

So there are three parameter types:  
* class generation probabilities `p(c|w)`  
* class membership probabilities `p(w|c)`  
* class n-gram probabilities `p(c_i|c_{i-n-1} .. c{i-1})`  

### Training morphologically motivated word classes

Below are example training commands for corpora in Finnish or Estonian.
In these examples the unknown word forms are tagged by the model with multiple classes per word.

#### Word classes for Finnish using Omorfi initialization

The initialization scripts under `omorfi` directory have been developed for the following Omorfi version  
`omorfi-analyse-text.sh -V`  
`omorfi-analyse-text 0.4 (Using omorfi bash API 20190511)`

Example training commands for a Utf-8 encoded training corpus `train.txt.gz` and an optional vocabulary file `vocab.txt`  
`zcat train.txt.gz |omorfi/omorfi_getwords.py -v vocab.txt >train.words.txt`  
`cat train.words.txt |omorfi-analyse-text.sh |gzip >words.analysis.gz`  
`cat train.words.txt |omorfi-analyse-text.sh -X |gzip >words.analysis.X.gz`  
`omorfi/omorfi_init_words.py words.analysis.gz words.analysis.X.gz words.omorfi.init words.omorfi.classes`  
`omorfi/omorfi_superclasses.py words.omorfi.classes words.omorfi.superclasses`

The expectation-maximization training can then be run as follows  
`scripts/trainer.py scripts/trainer.cfg words.omorfi.init train.txt.gz eval.txt.gz`


#### Word classes for Estonian using Estnltk initialization

The initialization scripts under `estnltk` directory have been developed for the Estnltk version 1.4.1  

Example training commands for a Utf-8 encoded training corpus `train.txt.gz`  
`estnltk/estnltk_init_words.py train.txt.gz words.estnltk.init words.estnltk.classes`  
`estnltk/estnltk_superclasses.py words.estnltk.classes words.estnltk.superclasses`


#### Refining morphological classes

The morphological classes trained from morphological analyses can be used to improve the classifications especially with smaller text corpora.
Below are the example commands for further processing the Finnish classes by first merging to 300 classes, then splitting to 1000 classes and finally running
the exchange algorithm with a constraint that exchanges are allowed only within the superclass created during the merge phase.

`merge train.txt.gz em.cmemprobs.gz words.omorfi.superclasses merge300 -c 300 -t 4`  
`split train.txt.gz merge300.cmemprobs.gz merge300.split1000 -c 1000`  
`zcat merge300.cmemprobs.gz |cut -f2 |cut -f1 -d' ' |sort -n |uniq >merge.classes`  
`scripts/update_superclasses.py merge300.cmemprobs.gz merge300.split1000.cmemprobs.gz`  
`exchange train.txt.gz exchange.1000.morphological -i merge300.split1000.cmemprobs.gz -t 4 -a 15`


## Programs in the package

### Model training

The expectation-maximization training programs write updated class generation probabilities, class membership probabilities
and alternative class n-gram sequences with probabilities to file. The class n-gram component may then be training using the
`ngram-count` program in the `SRILM` language modelling package. The `scripts/trainer.py` may be used for the full model training.
Example configuration is available in `scripts/trainer.cfg`.

* `init`          performs the 0th iteration for the expectation-maximization training
* `catstats`      runs one iteration of expectation-maximization training  

The following programs use bigram statistics for a model with one class per word

* `exchange`      exchange algorithm using bigram statistics
* `merge`         class merging using bigram statistics
* `split`         class splitting with local exchanges using bigram statistics

### Perplexity computation

* `ngramppl`      word n-gram perplexity
* `classppl`      class n-gram perplexity
* `classintppl`   perplexity for linear interpolation of a word n-gram model and a class n-gram model
* `catppl`        class n-gram perplexity for a model which allows multiple classes per word
* `catintppl`     perplexity for linear interpolation of a word n-gram model and a class n-gram model with multiple classes per word
