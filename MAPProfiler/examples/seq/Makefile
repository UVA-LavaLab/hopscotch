.PHONY: all clean

TARGET=seq

all: $(TARGET)

$(TARGET): $(TARGET).c
	gcc -g -O0 $(TARGET).c -o $(TARGET)

clean:
	rm -rf $(TARGET)
