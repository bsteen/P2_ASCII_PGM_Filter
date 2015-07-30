//CUDA Headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//C++ Headers
#include <iostream>
#include <ctime>
#include "MainH.h"
using namespace std;

__global__ void convolutionKernel(int * in_arr, int  * out_arr, int width, int height, float * stencil){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int p = row * grid_width + col;
	
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

__global__ void sobelKernel(int  * in_arr, int * out_arr, int const width, int const height, int * stencil){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int p = row * grid_width + col;

	int ul = (in_arr[p - width - 1]) * stencil[0];
	int um = (in_arr[p - width]) * stencil[1];
	int ur = (in_arr[p - width + 1]) * stencil[2];
	int ml = (in_arr[p - 1]) * stencil[3];
	int mm = (in_arr[p]) * stencil[4];
	int mr = (in_arr[p + 1]) * stencil[5];
	int ll = (in_arr[p + width - 1]) * stencil[6];
	int lm = (in_arr[p + width]) * stencil[7];
	int lr = (in_arr[p + width + 1]) * stencil[8];
	int x_sum = ul + um + ur + ml + mm + mr + ll + lm + lr;

	ul = (in_arr[p - width - 1]) * stencil[9];
	um = (in_arr[p - width]) * stencil[10];
	ur = (in_arr[p - width + 1]) * stencil[11];
	ml = (in_arr[p - 1]) * stencil[12];
	mm = (in_arr[p]) * stencil[13];
	mr = (in_arr[p + 1]) * stencil[14];
	ll = (in_arr[p + width - 1]) * stencil[15];
	lm = (in_arr[p + width]) * stencil[16];
	lr = (in_arr[p + width + 1]) * stencil[17];
	int y_sum = ul + um + ur + ml + mm + mr + ll + lm + lr;

	out_arr[p] = (int)pow((double)(y_sum*y_sum + x_sum*x_sum), 0.5);
}

//If CUDA function was no succesful, print out the error and exit.
void HANDLE_CUDA_ERROR(cudaError_t err, int line_num){
	if (err != cudaSuccess){
		printf("\nCUDA ERROR in %s:\n%s at line %i.\n\n", __FILE__, cudaGetErrorString(err), line_num);
		exit(EXIT_FAILURE);
	}
}

void launchKernel(int  * in_arr, int  * out_arr, int width, int height, char const filter_type, int filter_passes, double * time){
	//Check to see if there is a CUDA enabled device
	int count = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&count), __LINE__);
	
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	cout << "CUDA enabled device found: "<<props.name << " @" << props.clockRate/1000 << "Mhz" << endl;
	int major = props.major;
	cout << "Your device compute capability is: " << major << "." << props.minor << endl;
	if (major < 2){//A compute capability of 2.0 is needed to allow blocks with 1024 threads in them.
		errorExit("Your CUDA enabled device needs to have a compute capability of at least 2.0 to run the CUDA kernels.");
	}
	cout << "Applying Filter..." << endl;

	//Allocate device arrays
	int numElements = width * height;//width, height = orginal value + 2
	int * device_in_arr = 0;
	int * device_out_arr = 0;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_in_arr, sizeof(int)*numElements), __LINE__);
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_out_arr, sizeof(int)*numElements), __LINE__);

	//Copy both arrays to device memory
	HANDLE_CUDA_ERROR(cudaMemcpy(device_in_arr, in_arr, sizeof(int)*numElements, cudaMemcpyHostToDevice), __LINE__);
	HANDLE_CUDA_ERROR(cudaMemcpy(device_out_arr, out_arr, sizeof(int)*numElements, cudaMemcpyHostToDevice), __LINE__);

	int threadsPerBlock = 32;
	dim3 dimBlock(threadsPerBlock, threadsPerBlock);
	int blocksPerGrid = (int)ceil((double)numElements / (double)(threadsPerBlock*threadsPerBlock));//Guarantees that there are enough blocks in the grid for every element in the array.
	dim3 dimGrid(blocksPerGrid, 1);

	if (filter_type == '8'){//CUDA Box Blur
		//Allocated device memeory for the the device stencil and copy to device
		float boxblur_stencil[9] = { 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9 };
		float * device_stencil = 0;
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_stencil, sizeof(float) * 9), __LINE__);
		HANDLE_CUDA_ERROR(cudaMemcpy(device_stencil, boxblur_stencil, sizeof(float) * 9, cudaMemcpyHostToDevice), __LINE__);

		std::clock_t start = std::clock();
		for (int i = 0; i < filter_passes; i++){
			convolutionKernel << <dimGrid, dimBlock >> >(device_in_arr, device_out_arr, width, height, device_stencil);
			HANDLE_CUDA_ERROR(cudaGetLastError(), __LINE__);
			HANDLE_CUDA_ERROR(cudaMemcpy(device_in_arr, device_out_arr, sizeof(int)*numElements, cudaMemcpyDeviceToDevice), __LINE__);
		}
		*time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		HANDLE_CUDA_ERROR(cudaFree(device_stencil), __LINE__);
	}
	else if (filter_type == '9'){//CUDA Sobel Operator
		int  sobel_stencil[6][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 }, { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
		int * device_stencil = 0;
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_stencil, sizeof(int) * 18), __LINE__);
		HANDLE_CUDA_ERROR(cudaMemcpy(device_stencil, sobel_stencil, sizeof(int) * 18, cudaMemcpyHostToDevice), __LINE__);

		std::clock_t start = std::clock();
		for (int i = 0; i < filter_passes; i++){
			sobelKernel << <dimGrid, dimBlock >> >(device_in_arr, device_out_arr, width, height, device_stencil);
			HANDLE_CUDA_ERROR(cudaGetLastError(), __LINE__);
			HANDLE_CUDA_ERROR(cudaMemcpy(device_in_arr, device_out_arr, sizeof(int)*numElements, cudaMemcpyDeviceToDevice), __LINE__);
		}
		*time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		HANDLE_CUDA_ERROR(cudaFree(device_stencil), __LINE__);
	}
	//Copy filtered array out of device and back to host
	HANDLE_CUDA_ERROR(cudaMemcpy(out_arr, device_out_arr, sizeof(int)*numElements, cudaMemcpyDeviceToHost), __LINE__);

	//Free allocated device arrays and reset the device.
	HANDLE_CUDA_ERROR(cudaFree(device_in_arr), __LINE__);
	HANDLE_CUDA_ERROR(cudaFree(device_out_arr), __LINE__);
	cudaDeviceReset();
}