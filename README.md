# P2/ASCII PGM (Portable Gray Map) Image Filter
**This repository is for a research project.**

**About:**

This is an image processing program that uses CUDA. The program includes two methods of processing image data and applying filters:
-	Through normal C++ functions (Host/CPU, Serial Processing)
-	Through Nvidia’s CUDA kernels (Device/GPU, Parallel Processing)

This program takes in a valid P2 (ASCII Encoded) .PGM file and outputs a new .PGM file with the applied filter of choice. The main image filters featured in this program are the Sobel Operator/Filter and a Blur filter. Options are given to run a filter multiple times on the input image.
.PGM files must have dimensions **no greater than 1024x1024** (arbitrary size limit, I will change this later so that it accepts any arbitrary size).
There are still some weird bugs, but the program now works mostly as I intended it to.


**Compiling and Running:**

To compile this project, run make with the command: make

You can also import these project files into Microsoft Visual Studio (or any other IDE that supports CUDA) and compile it from there. Obviously, you must have the CUDA development kit and nvcc compiler installed as well (https://developer.nvidia.com/cuda-toolkit).

Your computer must also have a CUDA enabled NVIDIA GPU that has a compute compatibility of at least 2.0 in order to compile and/or run this program.

**Loading Valid .PGM Images:**

Image files to be read in are placed in the "Images" sub-folder, and filtered images appear in the "Output" sub-folder with "_output.pgm" appended to their file name.
You can convert other image formats to .PGM with free imaging software such as IrfanView (http://www.irfanview.com/) or GIMP (http://www.gimp.org/). Make sure you select the ASCII encoding option if you want the files to be accepted by this program.

Some of the sample P2 .PGM files used in this project were obtained at: http://people.sc.fsu.edu/~jburkardt/data/pgma/pgma.html


**Loading Methods:**

To load an image, there are two methods available. The first (enabled by default) has the program ask you to type in the name of the image file. This file must be located in the "Images" subfolder. This first method is internally labeled as manualLoadImage. The second method can be enabled by commenting out the call to the manualLoadImage method and uncommenting the line that calls directLoadImage. The program then needs to be recompiled. Once this is done, images are now loaded with this bash syntax: "./filter.out < /path/to/image.pgm" After running this command, the program will automatically load the piped in image file, without giving you a prompt to enter the file name. Both of these methods accomplish the same thing, but the directLoadImage allows you to pipe in image files that are not placed within the Images sub folder. Either way, the final image is placed into the Output subfolder.


**Planned Features:**

Remove the arbitrary image size limit. Do do this I will replace the main data array with a vector. I will need to use CUDA specific vectors instead of the standard C++ STL ones in order for them to be passed to the kernels. http://docs.nvidia.com/cuda/thrust/index.html#vectors

Currently, both methods of processing an image (CPU/Serial and GPU/CUDA/Parallel) pass through the image data in a linear fashion. In the future, I will implement a Fast Fourier Transformation (FFT) using CUDA's FFT library. This will hopefully increase the CUDA filter preformance even more. I might add a FFT method for the CPU/Serial filters, but they would not be using the CUDA FFT library.

I also hope to add the ability to display loaded images from within the program. Using OpenGL, you would be able to display an unfiltered image, run various filters on the image, and see the resutls diplayed in real time from within the program. This won't be added for a long time, and it will only be added after the FFT is implemeted.
