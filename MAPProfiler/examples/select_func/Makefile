.PHONY: all clean

TARGET=select_func

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	g++ -std=c++11 -g -O0 $(TARGET).cpp -o $(TARGET)

clean:
	rm -rf $(TARGET)
