NVCC = nvcc
NVCCFLAGS =
HEADERS = cuda_check.hpp
SOURCES = main.cu cuda_check.cu
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
