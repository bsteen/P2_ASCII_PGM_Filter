# P2/ASCII PGM (Portable Gray Map) Image Filter
**This repository is for a research project.**

An image processing program that uses CUDA. This program includes two methods of processing image data and applying filters:
-	Through normal C++ functions (CPU/Serial Processing)
-	Through Nvidia’s CUDA kernels (GPU/Parallel Processing)

This program takes in a valid P2 (ASCII Encoded) .PGM file and outputs a new .PGM file with the applied filter of choice. The main image filters featured in this program are the Sobel Operator/Filter and a Blur filter. Options are given to run a filter multiple times on the input image.
.PGM files must have dimensions no greater than 1024x1024 (arbitrary size limit, I will change this later so that it accepts any arbitrary size).

There are still some weird bugs, but it works mostly as I intended to.
To complile this project, run in a Linux terminal: nvcc Main.cu Kernel.cu -std=c++11

You can also import these files into Microsoft Visual Studio (or any other IDE that supports CUDA) and compile it from there. Obviously, you must have the CUDA development kit and nvcc compiler installed as well (https://developer.nvidia.com/cuda-toolkit).

Image files to be read in are placed in the "Images" sub-folder, and filtered images appear in the "Output" sub-folder with "_output.pgm" appended to their file name.
You can convert other image formats to .PGM with free image software such as IrfanView (http://www.irfanview.com/) or GIMP (http://www.gimp.org/). Make sure you select the ASCII encoding option if you want the files to be accepted by this program.

Your computer must also have a CUDA enabled NVIDIA GPU that has a compute compatibility of at least 2.0 in order to compile and/or run the CUDA filters.

Some of the sample P2 .PGM files used in this project were obtained at: http://people.sc.fsu.edu/~jburkardt/data/pgma/pgma.html
