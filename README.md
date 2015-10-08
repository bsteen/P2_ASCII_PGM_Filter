# P2/ASCII PGM (Portable Gray Map) Image Filter
**This repository is for a research project.**

An image processing program that uses CUDA. This program includes two methods of processing image data and applying filters:
-	Through normal C++ functions (Host/CPU, Serial Processing)
-	Through Nvidia’s CUDA kernels (Device/GPU, Parallel Processing)

This program takes in a valid P2 (ASCII Encoded) .PGM file and outputs a new .PGM file with the applied filter of choice. The main image filters featured in this program are the Sobel Operator/Filter and a Blur filter. Options are given to run a filter multiple times on the input image.
.PGM files must have dimensions no greater than 1024x1024 (arbitrary size limit, I will change this later so that it accepts any arbitrary size).

There are still some weird bugs, but it works mostly as I intended to.
To compile this project, run make with the command: make

You can also import these files into Microsoft Visual Studio (or any other IDE that supports CUDA) and compile it from there. Obviously, you must have the CUDA development kit and nvcc compiler installed as well (https://developer.nvidia.com/cuda-toolkit).

Image files to be read in are placed in the "Images" sub-folder, and filtered images appear in the "Output" sub-folder with "_output.pgm" appended to their file name.
You can convert other image formats to .PGM with free image software such as IrfanView (http://www.irfanview.com/) or GIMP (http://www.gimp.org/). Make sure you select the ASCII encoding option if you want the files to be accepted by this program.

To load an image, there are two methods available. The first (enabled by default) has the program ask you to type in the name of the image file. This file must be located in the "Images" subfolder. This first method is internally labeled as manualLoadImage. The second method can be enabled by commenting out the call to the manualLoadImage method and uncommenting the line that calls directLoadImage. The program then needs to be recompiled. Once this is done, images are now loaded with this bash syntax: "./filter.out < /path/to/image.pgm" After running this command, the program will automatically load the piped in image file, without giving you a prompt to enter the file name. Both of these methods accomplish the same thing, but the directLoadImage allows you to pipe in image files that are not placed within the Images sub folder. Either way, the final image is placed into the Output subfolder.

Your computer must have a CUDA enabled NVIDIA GPU that has a compute compatibility of at least 2.0 in order to compile and/or run this program.

Some of the sample P2 .PGM files used in this project were obtained at: http://people.sc.fsu.edu/~jburkardt/data/pgma/pgma.html
