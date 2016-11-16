cxxflags = -O3 -DNDEBUG -std=gnu++0x -Wall -Wno-unused-function
#cxxflags = -O0 -g -std=gnu++0x -Wall -Wno-unused-function

##################################################

progs = init catstats ngramppl classppl classintppl catppl catintppl catintppl2 exchange merge split
progs_srcs = $(progs:=.cc)
progs_objs = $(progs:=.o)
srcs = conf.cc io.cc Categories.cc Ngram.cc CatPerplexity.cc ExchangeAlgorithm.cc
objs = $(srcs:.cc=.o)

test_progs = runtests
test_progs_srcs = $(test_progs:=.cc)
test_progs_objs = $(test_progs:=.o)
test_srcs = ppltest.cc categorytest.cc exchangetest.cc mergetest.cc
test_objs = $(test_srcs:.cc=.o)

##################################################

.SUFFIXES:

all: $(progs) $(test_progs)

%.o: %.cc
	$(CXX) -c $(cxxflags) $< -o $@

$(progs): %: %.o $(objs)
	$(CXX) $(cxxflags) $< -o $@ $(objs) -lz -pthread

%: %.o $(objs)
	$(CXX) $(cxxflags) $< -o $@ $(objs)

$(test_progs): %: %.o $(objs) $(test_objs)
	$(CXX) $(cxxflags) $< -o $@ $(objs) $(test_objs) -lboost_unit_test_framework -lz -pthread

test_objs: $(test_srcs)

test_progs: $(objs) $(test_objs)

.PHONY: clean
clean:
	rm -f $(objs) *.o $(progs) $(progs_objs) .depend *~ *.exe

dep:
	$(CXX) -MM $(cxxflags) $(DEPFLAGS) $(all_srcs) > dep
include dep

