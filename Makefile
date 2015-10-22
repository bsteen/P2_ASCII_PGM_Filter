compile: Main.o Kernel.o Load.o
	nvcc obj/Main.o obj/Kernel.o obj/Load.o -o filter.out -std=c++11

run: compile
	clear
	./filter.out

Main.o: Main.cu
	nvcc -c Main.cu -o obj/Main.o -std=c++11

Kernel.o: Kernel.cu
	nvcc -c Kernel.cu -o obj/Kernel.o -std=c++11

Load.o: Load.cu
	nvcc -c Load.cu -o obj/Load.o -std=c++11

clean: #Remove object files and executable
	rm obj/*o filter.out
	clear
