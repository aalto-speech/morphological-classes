-include Makefile.local

cxxflags += -std=gnu++0x

##################################################

progs = init\
	catstats\
	ngramppl\
	classppl\
	classintppl\
	catppl\
	catintppl\
	swngramppl\
	exchange\
	merge\
	split
progs_srcs = $(addsuffix .cc,$(addprefix src/,$(progs)))
progs_objs = $(addsuffix .o,$(addprefix src/,$(progs)))

srcs = util/conf.cc\
	util/io.cc\
	src/Categories.cc\
	src/Ngram.cc\
	src/CatPerplexity.cc\
	src/Exchanging.cc\
	src/Merging.cc\
	src/Splitting.cc\
	src/ModelWrappers.cc
objs = $(srcs:.cc=.o)

ifndef NO_UNIT_TESTS
test_progs = runtests
test_progs_srcs = $(test_progs:=.cc)
test_progs_objs = $(test_progs:=.o)
test_srcs = test/ppltest.cc\
	test/categorytest.cc\
	test/exchangetest.cc\
	test/mergetest.cc\
	test/splittest.cc
test_objs = $(test_srcs:.cc=.o)
endif

##################################################

.SUFFIXES:

all: $(progs) $(test_progs)

%.o: %.cc
	$(CXX) -c $(cxxflags) $< -o $@ -I./util

$(progs): $(progs_objs) $(objs)
	$(CXX) $(cxxflags) -o $@ src/$@.cc $(objs) -lz -pthread -I./util -I./src

%: %.o $(objs)
	$(CXX) $(cxxflags) $< -o $@ $(objs)

$(test_progs): $(test_objs)
	$(CXX) $(cxxflags) -o $@ test/$@.cc $(objs) $(test_objs)\
	 -lboost_unit_test_framework -lz -pthread -I./util -I./src

$(test_objs): %.o: %.cc $(objs)
	$(CXX) -c $(cxxflags) $< -o $@ -I./util -I./src

.PHONY: clean
clean:
	rm -f $(objs) $(progs_objs) $(test_objs)\
	 $(progs) $(test_progs)  .depend *~ *.exe

dep:
	$(CXX) -MM $(cxxflags) $(DEPFLAGS) $(all_srcs) > dep
include dep

