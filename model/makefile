NVCC = nvcc
NVCCFLAGS =
SOURCES = main.cu
TARGET = main

.PHONY: clean check

$(TARGET): $(SOURCES)
	$(NVCC) -o $@ $^ $(NVCCFLAGS)

check:
	cuda-memcheck ./$(TARGET)
	cuda-memcheck --tool racecheck ./$(TARGET)
	cuda-memcheck --tool initcheck ./$(TARGET)
	cuda-memcheck --tool synccheck ./$(TARGET)

clean:
	-rm -f $(TARGET)
