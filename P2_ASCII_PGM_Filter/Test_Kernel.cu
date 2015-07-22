//CUDA Headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "KernelH.cuh"
//C++ Headers
#include <iostream>
using namespace std;

__global__ void kernel(int * in_arr, int * out_arr, int width, int height, float * stencil){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int p = row*width + col;

	if (row>0 && row <height - 1 && col>0 && col<width - 1){
		float ul = (float)(in_arr[p - width - 1]) * stencil[0];
		float um = (float)(in_arr[p - width]) * stencil[1];
		float ur = (float)(in_arr[p - width + 1]) * stencil[2];

		float ml = (float)(in_arr[p - 1]) * stencil[3];
		float mm = (float)(in_arr[p]) *stencil[4];
		float mr = (float)(in_arr[p + 1]) * stencil[5];

		float ll = (float)(in_arr[p + width - 1]) * stencil[6];
		float lm = (float)(in_arr[p + width]) * stencil[7];
		float lr = (float)(in_arr[p + width + 1]) * stencil[8];

		out_arr[p] = (int)(ul + um + ur + ml + mm + mr + ll + lm + lr);
	}
	else{
		out_arr[p] = 0;
	}
	
}

__host__ void prepareTest(){
	int const width = 5;
	int const height = 5;
	int const numElements = width*height;

	int host_in_arr[numElements] = { 0, 0, 0, 0, 0,
									 0, 100, 100, 100, 0,
									 0, 100, 100, 100, 0, 
									 0, 100, 100, 100, 0, 
									 0, 0, 0, 0, 0};
	int host_out_arr[numElements];

	int * dev_in_arr = 0;
	int * dev_out_arr = 0;

	printCUDAError(cudaMalloc(&dev_in_arr, sizeof(int)*numElements), __LINE__);
	printCUDAError(cudaMemcpy(dev_in_arr, host_in_arr, sizeof(int)*numElements, cudaMemcpyHostToDevice), __LINE__);
	printCUDAError(cudaMalloc(&dev_out_arr, sizeof(int)*numElements), __LINE__);
	printCUDAError(cudaMemcpy(dev_out_arr, host_out_arr, sizeof(int)*numElements, cudaMemcpyHostToDevice), __LINE__);

	double threadsPerBlock = 16.0;
	int blocksPerGrid = (int)ceil((double)numElements / threadsPerBlock);//Guarantees that there are enough blocks in the grid for every element in the array.
	dim3 dimBlock((int)threadsPerBlock, (int)threadsPerBlock); //16x16 thread blocks
	dim3 dimGrid(blocksPerGrid, blocksPerGrid);

	float boxblur_stencil[9] = { 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9 };
	float * device_stencil;
	printCUDAError(cudaMalloc(&device_stencil, sizeof(float) * 9), __LINE__);
	printCUDAError(cudaMemcpy(device_stencil, boxblur_stencil, sizeof(float) * 9, cudaMemcpyHostToDevice), __LINE__);

	kernel << <dimGrid, dimBlock >> >(dev_in_arr, dev_out_arr, width, height, device_stencil);

	printCUDAError(cudaMemcpy(host_out_arr, dev_out_arr, sizeof(int)*numElements, cudaMemcpyDeviceToHost), __LINE__);

	for (int r = 0; r < height; r++){
		for (int c = 0; c < width; c++){
			cout << host_out_arr[r*width+c] << " ";
		}
		cout << endl;
	}
}