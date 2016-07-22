

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =


include ../config.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES =


OBJFILES = cuda-device.o cuda-math.o cuda-matrix.o cuda-vector.o cuda-common.o cuda-rand.o
ifeq ($(CUDA), true)
  OBJFILES += cuda-kernels.o cuda-randkernels.o
endif

LIBNAME = gpucompute

all:  $(LIBFILE)

ifeq ($(CUDA_ARCH),) 
  $(error %%% Missing CUDA_ARCH in 'config.mk')
endif

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../


ADDLIBS = ../cpucompute/cpucompute.a ../base/base.a  ../util/util.a 

include ../makefiles/default_rules.mk

