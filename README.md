# P2/ASCII PGM (Portable Gray Map) Image Filter
An image processing program that uses C++ and CUDA. This program includes two methods of processing image data and applying filters:
-	Through C++ functions (Serial Processing)
-	Through Nvidia’s CUDA kernels (Parallel Processing)

This program takes in a valid P2 (ASCII Encoded) .PGM file and outputs a new .PGM file with the applied filter of choice. The main image filters featured in this program are the Sobel Operator/Filter and a Blur filter.
.PGM files must have dimensions no greater than 1024x1024 (arbitrary limit).
