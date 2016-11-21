# P2/ASCII PGM (Portable Gray Map) Image Filter

<p align="center">
  <img src="https://github.com/bsteen/P2_ASCII_PGM_Filter/blob/master/sample.gif" alt="Sample filtered images"/>
</p>

*Sample images shown above are scaled down from 512x512 to 300x300.

**About:**

This is an image processing program that uses CUDA. The program includes two methods of processing image data and applying filters:
-	Through normal C++ functions (Host/CPU, Serial Processing)
-	Through Nvidia’s CUDA kernels (Device/GPU, Parallel Processing)

This program takes in a valid P2 (ASCII Encoded) .PGM file and outputs a new .PGM file with the applied filter of choice. The main image filters featured in this program are the Sobel Operator/Filter and a Blur filter. Options are given to run a filter multiple times on the input image.
.PGM files must have dimensions **no greater than 1024x1024** (arbitrary size limit, I will change this later so that it accepts any arbitrary size).
The final image is then displayed using OpenGL and SFML.
There are still some weird bugs, but the program now works mostly as I intended it to.

**Purpose:**

The purpose of this project is to act as learning experience and introduce me to programming kernels in CUDA. When I started making this program, I knew nothing about
C, C++, CUDA, or parallel programing. As of now, I feel much more comfortable with C and C++, as well as the concepts of programing. CUDA is still difficult for me, but
I am currently taking a class about parallel programing.

**Compiling and Running:**

To compile this project, run make with the command: make

You can also import these project files into Microsoft Visual Studio (or any other IDE that supports CUDA) and compile it from there. You must have the CUDA development kit and nvcc compiler installed as well (https://developer.nvidia.com/cuda-toolkit).

Your computer must also have a CUDA enabled NVIDIA GPU that has a compute compatibility of at least 2.0 in order to compile andor run this program.
Because of the image display methods, the SFML library(http://www.sfml-dev.org/) and GLEW(http://glew.sourceforge.net/) must also be installed.

**Loading Valid .PGM Images:**

Image files to be read in are given as command line arguments, and filtered images appear in the "Output" sub-folder with "_output.pgm" appended to their file name.
You can convert other image formats to .PGM with free imaging software such as IrfanView (http://www.irfanview.com/) or GIMP (http://www.gimp.org/). Make sure you select the ASCII encoding option if you want the files to be accepted by this program.

Some of the sample P2 .PGM files used in this project were obtained at: http://people.sc.fsu.edu/~jburkardt/data/pgma/pgma.html

**Planned Features and Improvements/Fixes:**

Clean up the code know that I know C++ better.

Remove the arbitrary image size limit. To do this, I will replace the main data array with a vector. I will need to use CUDA specific vectors instead of the C++ STL ones in order for them to be passed to the kernels. http://docs.nvidia.com/cuda/thrust/index.html#vectors

Currently, both methods of processing an image (CPU/Serial and GPU/CUDA/Parallel) pass through the image data in a linear fashion. In the future, I will implement a Fast Fourier Transformation (FFT) using CUDA's FFT library. This will hopefully increase the CUDA filter performance even more. I might add a FFT method for the CPU/Serial filters, but they would not be using the CUDA FFT library as I want them to only use the CPU. This is done for performance comparison sake (serial vs parallel). This will be done much later, after the fixes and OpenGL stuff is added.
