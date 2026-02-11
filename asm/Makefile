# C wrapper for Assembly kernels
# macOS compatible

CC = gcc
NASM = nasm
CFLAGS = -O3 -march=native -fPIC -shared -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/include/python3.9 -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/include
NASMFLAGS = -f macho64

all: libacie_asm.dylib

matrix_kernels.o: matrix_kernels.asm
	$(NASM) $(NASMFLAGS) $< -o $@

acie_asm_wrapper.o: acie_asm_wrapper.c
	$(CC) $(CFLAGS) -c $< -o $@

libacie_asm.dylib: matrix_kernels.o acie_asm_wrapper.o
	$(CC) -dynamiclib -o $@ $^ -L/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib -lpython3.9

clean:
	rm -f *.o *.dylib *.so

.PHONY: all clean
