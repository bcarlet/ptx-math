NVCC = nvcc
NVCCFLAGS = -std=c++11 -O2
HEADERS = *.hpp
SOURCES = $(wildcard *.cu)
TARGET = main

.PHONY: clean check

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) -o $@ $(SOURCES) $(NVCCFLAGS)

check:
	cuda-memcheck ./$(TARGET)
	cuda-memcheck --tool racecheck ./$(TARGET)
	cuda-memcheck --tool initcheck ./$(TARGET)
	cuda-memcheck --tool synccheck ./$(TARGET)

clean:
	$(RM) $(TARGET)
