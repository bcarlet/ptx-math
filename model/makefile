NVCC = nvcc
NVCCFLAGS = -std=c++11 -O2

HEADERS = \
	cuda_util.hpp ptx.hpp util/stopwatch.hpp util/tally.hpp \
	util/progress.hpp model/model.h model/fpassert.h model/fpmanip.h \
	model/nan.h
SOURCES = \
	main.cu cuda_util.cu util/progress.cpp model/fpmanip.c model/nan.c \
	model/rcp.c model/sqrt.c model/rsqrt.c model/sin.c model/cos.c \
	model/lg2.c model/ex2.c

TARGET = main

.PHONY: clean

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) -o $@ $(SOURCES) $(NVCCFLAGS)

clean:
	$(RM) $(TARGET)
