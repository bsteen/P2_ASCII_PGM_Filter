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

		out_arr[p] = in_arr[p] + 1;
	}
	else{
		out_arr[p] = 0;
	}
	
}

__host__ void prepareTest(int * in_arr){
	int  width = 5;
	int  height = 5;
	int const numElements = width*height;

	int * host_in_arr = (int*)malloc(sizeof(int)*numElements);
	int * host_out_arr= (int*)malloc(sizeof(int)*numElements);

	host_in_arr[0] = host_in_arr[1] = host_in_arr[2] = host_in_arr[3] = host_in_arr[4] = 0;
	host_in_arr[5] = 0; host_in_arr[6] = host_in_arr[7] = host_in_arr[8] = 1;	host_in_arr[9] = 0;
	host_in_arr[10] = 0; host_in_arr[11] = host_in_arr[12] = host_in_arr[13] = 1;	host_in_arr[14] = 0;
	host_in_arr[15] = 0; host_in_arr[16] = host_in_arr[17] = host_in_arr[18] = 1;	host_in_arr[19] = 0;
	host_in_arr[20] = host_in_arr[21] = host_in_arr[22] = host_in_arr[23] = host_in_arr[24] = 0;

	int * dev_in_arr = 0;
	int * dev_out_arr = 0;

	HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_in_arr, sizeof(int)*numElements), __LINE__);
	HANDLE_CUDA_ERROR(cudaMemcpy(dev_in_arr, host_in_arr, sizeof(int)*numElements, cudaMemcpyHostToDevice), __LINE__);
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_out_arr, sizeof(int)*numElements), __LINE__);
	HANDLE_CUDA_ERROR(cudaMemcpy(dev_out_arr, host_out_arr, sizeof(int)*numElements, cudaMemcpyHostToDevice), __LINE__);

	double threadsPerBlock = 16.0;
	int blocksPerGrid = (int)ceil((double)numElements / threadsPerBlock);//Guarantees that there are enough blocks in the grid for every element in the array.
	dim3 dimBlock((int)threadsPerBlock, (int)threadsPerBlock); //16x16 thread blocks
	dim3 dimGrid(blocksPerGrid, blocksPerGrid);

	float boxblur_stencil[9] = { 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9 };
	float * device_stencil;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_stencil, sizeof(float) * 9), __LINE__);
	HANDLE_CUDA_ERROR(cudaMemcpy(device_stencil, boxblur_stencil, sizeof(float) * 9, cudaMemcpyHostToDevice), __LINE__);

	kernel << <dimGrid, dimBlock >> >(dev_in_arr, dev_out_arr, width, height, device_stencil);

	HANDLE_CUDA_ERROR(cudaMemcpy(host_out_arr, dev_out_arr, sizeof(int)*numElements, cudaMemcpyDeviceToHost), __LINE__);

	for (int r = 0; r < height; r++){
		for (int c = 0; c < width; c++){
			cout << host_out_arr[r*width+c] << " ";
		}
		cout << endl;
	}
}