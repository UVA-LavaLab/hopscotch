.PHONY: all clean

TARGET=scatter_gather

all: $(TARGET)

$(TARGET): $(TARGET).c
	gcc -g -O0 $(TARGET).c -o $(TARGET)

clean:
	rm -rf $(TARGET)
