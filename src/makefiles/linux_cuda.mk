CUDA_ARCH = -gencode arch=compute_20,code=sm_20 \
	    -gencode arch=compute_30,code=sm_30 \
	    -gencode arch=compute_35,code=sm_35 \
	    -gencode arch=compute_52,code=sm_52 \
	    -gencode arch=compute_61,code=sm_61   # new in CUDA 8.0,

CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 32 -DHAVE_CUDA

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include 
LDFLAGS += -L$(CUDATKDIR)/lib -Wl,-rpath=$(CUDATKDIR)/lib
LDLIBS += -lcublas -lcudart #LDLIBS : The libs are loaded later than static libs in implicit rule

