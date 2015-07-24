//CUDA Headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "KernelH.cuh"
//C++ Headers
#include <iostream>
using namespace std;

__global__ void kernel(int * in_arr, int  * out_arr, int width, int height, float * stencil){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int grid_width = gridDim.x * blockDim.x;
	int p = row * grid_width + col;

	if (row > 0 && row < height - 1 && col>0 && col < width - 1){
		out_arr[p] = in_arr[p];
	}
	else{
		out_arr[p] = 0;
	}
}

void launchTestKernel(int  * in_arr, int  * out_arr, int width, int height, char filter_type){
	int const numElements = width*height;
	int * device_in_arr;
	int * device_out_arr;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_in_arr, sizeof(int)*numElements), __LINE__);
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_out_arr, sizeof(int)*numElements), __LINE__);

	float * device_stencil = 0;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_stencil, sizeof(float) * 9), __LINE__);

	int threadsPerBlock = 32;
	dim3 dimBlock(threadsPerBlock, threadsPerBlock);
	int blocksPerGrid = (int)ceil((double)numElements / (double)(threadsPerBlock*threadsPerBlock));//Guarantees that there are enough blocks in the grid for every element in the array.
	dim3 dimGrid(blocksPerGrid, 1);

	kernel << <dimGrid, dimBlock>> >(device_in_arr, device_out_arr, width, height, device_stencil);
	HANDLE_CUDA_ERROR(cudaDeviceSynchronize(), __LINE__);

	HANDLE_CUDA_ERROR(cudaMemcpy(out_arr, device_out_arr, sizeof(int)*numElements, cudaMemcpyDeviceToHost), __LINE__);

	HANDLE_CUDA_ERROR(cudaFree(device_stencil), __LINE__);
	HANDLE_CUDA_ERROR(cudaFree(device_in_arr), __LINE__);
	HANDLE_CUDA_ERROR(cudaFree(device_out_arr), __LINE__);
	HANDLE_CUDA_ERROR(cudaDeviceReset(), __LINE__);
}