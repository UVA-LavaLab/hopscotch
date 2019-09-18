.PHONY: all clean

TARGET=omp

all: $(TARGET)

$(TARGET): $(TARGET).c
	gcc -g -O0 $(TARGET).c -o $(TARGET) -fopenmp

clean:
	rm -rf $(TARGET)
