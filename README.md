# P2/ASCII PGM (Portable Gray Map) Image Filter
An image processing program that uses C++ and CUDA. This program includes two methods of processing image data and applying filters:
-	Through C++ functions (Serial Processing)
-	Through Nvidia’s CUDA kernels (Parallel Processing)

This program takes in a valid P2 (ASCII Encoded) .PGM file and outputs a new .PGM file with the applied filter of choice. The main image filters featured in this program are the Sobel Operator/Filter and a Blur filter.
.PGM files must have dimensions no greater than 1024x1024 (arbitrary limit).

There are still some weird bugs, but it works mostly as I intended to be. This repository is made up of a Microsoft Visual Studio project. All relevant files are in the main "P2_ASCII_PGM_Filter" directory.

Image files to be read in are placed in the "Images" sub-folder, and filtered images appear in the "Output" sub-folder with "_output.pgm" appended to their file name.
You can convert other image formats to .PGM with free image software such as IrfanView (http://www.irfanview.com/) or GIMP (http://www.gimp.org/). Make sure you select the ASCII encoding option if you want the files to be accepted by this program.

Your computer must also have a CUDA enabled NVIDIA GPU that has compute compatibility of at least 2.0 in order to run the CUDA filters.

Some of the sample P2 .PGM files used in this project were obtained at: http://people.sc.fsu.edu/~jburkardt/data/pgma/pgma.html
