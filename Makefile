compile: Main.o Kernel.o Load.o
	nvcc -o filter.out obj/Main.o obj/Kernel.o obj/Load.o Window.cpp -lsfml-window -lsfml-system -lGLEW -lGL -std=c++11
run:
	clear
	./filter.out

compile-run: compile run

Main.o:
	nvcc -c Main.cu -o obj/Main.o -std=c++11

Load.o:
	nvcc -c Load.cu -o obj/Load.o -std=c++11

Kernel.o:
	nvcc -c Kernel.cu -o obj/Kernel.o -std=c++11

clean: #Remove object files and executable
	rm obj/*o filter.out
	clear
