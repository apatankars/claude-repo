CC = nvcc -ccbin clang++
CCFLAGS = -O3 --compiler-options="-g3 -Wall -Wextra -Werror"
# add debugging information, at the cost of speed
# CCFLAGS += -g -G
# get verbose output from the PTX optimizing assembler
# CCFLAGS += --ptxas-options=-v

EXECS = saxpy sgemm conv1d

.PHONY: all clean check

all: $(EXECS)

saxpy: vector_addition/saxpy.cu
	$(CC) $(CCFLAGS) $^ -o $@

sgemm: matrix_multiply/sgemm.cu matrix_multiply/kernels.cuh
	$(CC) $(CCFLAGS) $< -o $@ -lcublas

conv1d: depth_conv1d/conv1d_cuda.cu depth_conv1d/conv1d_kernel.cuh
	$(CC) $(CCFLAGS) $< -o $@

# See here for more: https://stackoverflow.com/a/14061796.
ifeq (check,$(firstword $(MAKECMDGOALS))) # if the first argument is "check",
  # use the rest as arguments for "check", and turn into do-nothing targets
  CHECK_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(CHECK_ARGS):;@:)
endif

check: all
	@echo "python3 run_tests.py $(CHECK_ARGS)"
	@bash -c "source ~/cs1390-venv/bin/activate && python3 run_tests.py $(CHECK_ARGS)"

clean:
	rm -f $(EXECS)