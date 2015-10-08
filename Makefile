compile: Main.o Kernel.o Load.o
	nvcc Main.o Kernel.o Load.o -o filter.out -std=c++11

Main.o: Main.cu
	nvcc -c Main.cu -std=c++11

Kernel.o: Kernel.cu
	nvcc -c Kernel.cu -std=c++11

Load.o: Load.cu
	nvcc -c Load.cu -std=c++11

clean: #Remove object files and executable
	rm *o filter.out
