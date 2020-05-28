NVCC = nvcc
NVCCFLAGS = -std=c++11 -O2

HEADERS = \
	cuda_util.hpp ptx.hpp util/stopwatch.hpp util/tally.hpp \
	util/progress.hpp model/model.h
SOURCES = \
	main.cu cuda_util.cpp util/progress.cpp model/model.c

TARGET = main

.PHONY: clean

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) -o $@ $(SOURCES) $(NVCCFLAGS)

clean:
	$(RM) $(TARGET)
