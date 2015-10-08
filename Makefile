compile: Main.o Kernel.o
	nvcc Main.cu Kernel.cu -o filter.out -std=c++11

Main.o: Main.cu
	nvcc -c Main.cu -std=c++11

Kernel.o: Kernel.cu
	nvcc -c Kernel.cu -std=c++11

clean: #Remove object files and executable
	rm *o filter.out
