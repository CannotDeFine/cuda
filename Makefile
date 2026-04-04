NVCCFLAGS := -g --use_fast_math
OPENGL_LIBS := -lglut -lGL -lGLU
SRC_DIRS := chapter03 chapter04 chapter05 chapter06

ALL_SRCS := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cu))
EMPTY_SRCS := $(foreach src,$(ALL_SRCS),$(if $(shell test ! -s $(src) && echo yes),$(src),))
SKIP_SRCS := 
SRCS := $(filter-out $(EMPTY_SRCS) $(SKIP_SRCS),$(ALL_SRCS))
BINS := $(patsubst %.cu,bin/%,$(SRCS))

all: $(BINS)

bin/chapter05/animate: EXTRA_LIBS := $(OPENGL_LIBS)
bin/chapter05/bitmap: EXTRA_LIBS := $(OPENGL_LIBS)
bin/chapter06/ray_tracing: EXTRA_LIBS := $(OPENGL_LIBS)
bin/chapter06/ray_tracing_with_const: EXTRA_LIBS := $(OPENGL_LIBS)

bin/%: %.cu
	@mkdir -p $(dir $@)
	nvcc $(NVCCFLAGS) $< -o $@ $(EXTRA_LIBS)

compdb:
	bear -- make clean all

clean:
	rm -f $(BINS)
