.PHONY: all query clean roofline

all: query

query: common.cu common.h query.cu
	nvcc common.cu query.cu -o query

roofline: common.cu common.h roofline.cu
	nvcc common.cu roofline.cu -o roofline -I../../../include/ -std=c++11 $(USER_DEFS)

clean:
	@rm -rf query
	
