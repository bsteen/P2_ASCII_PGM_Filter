//CUDA Headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
//C++ Headers
#include <iostream>
using namespace std;

__global__ void convolutionKernel(int * in_arr, int  * out_arr, int width, int height, float * stencil){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	int grid_width = gridDim.x * blockDim.x;
	int p = row * grid_width + col;

	if (row>0 && row <height - 1 && col>0 && col<width - 1){//Only pass stencil over pixels inside the border
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
	else{//Border values should just be set to zero.
		out_arr[p] = 0;
	}
}

/*This kernel won't be implements until the convolutionKernel works.
__global__ void sobelKernel(int  * in_arr, int * out_arr, int const width, int const height, int * stencil){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row > 0 && row < height - 1 && col>0 && col < width - 1){//bad, need to find another way
		int p = row*width + col;//position

		int ul = (in_arr[p - width - 1]) * stencil[0][0];
		int um = (in_arr[p - width]) * stencil[0][1];
		int ur = (in_arr[p - width + 1]) * stencil[0][2];
		int ml = (in_arr[p - 1]) * stencil[1][0];
		int mm = (in_arr[p]) * stencil[1][1];
		int mr = (in_arr[p + 1]) * stencil[1][2];
		int ll = (in_arr[p + width - 1]) * stencil[2][0];
		int lm = (in_arr[p + width]) * stencil[2][1];
		int lr = (in_arr[p + width + 1]) * stencil[2][2];
		int x_sum = ul + um + ur + ml + mm + mr + ll + lm + lr;

		ul = (in_arr[p - width - 1]) * stencil[3][0];
		um = (in_arr[p - width]) * stencil[3][1];
		ur = (in_arr[p - width + 1]) * stencil[3][2];
		ml = (in_arr[p - 1]) * stencil[4][0];
		mm = (in_arr[p]) * stencil[4][1];
		mr = (in_arr[p + 1]) * stencil[4][2];
		ll = (in_arr[p + width - 1]) * stencil[5][0];
		lm = (in_arr[p + width]) * stencil[5][1];
		lr = (in_arr[p + width + 1]) * stencil[5][2];
		int y_sum = ul + um + ur + ml + mm + mr + ll + lm + lr;

		out_arr[p] = (int)pow((double)(y_sum*y_sum + x_sum*x_sum), 0.5);
	}
}*/

//If CUDA function was no succesful, print out the error and exit.
__host__ void HANDLE_CUDA_ERROR(cudaError_t err, int line_num){
	if (err != cudaSuccess){
		printf("\nCUDA ERROR in %s:\n%s at line %i.\n\n", __FILE__, cudaGetErrorString(err), line_num);
		exit(EXIT_FAILURE);
	}
}

__host__ void launchKernel(int const  * in_arr, int  * out_arr, int width, int height, char const filter_type){
	//Check to see if there is a CUDA enabled device
	int count = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&count), __LINE__);
	
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	cout << "CUDA enabled device found: "<<props.name << " @" << props.clockRate/1000 << "Mhz" << endl;
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

	//Allocated device memeory for the the device stencil
	float * device_stencil = 0;
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_stencil, sizeof(float) * 9), __LINE__);

	int threadsPerBlock = 16;
	int blocksPerGrid = (int)ceil((double)numElements / (double)threadsPerBlock);//Guarantees that there are enough blocks in the grid for every element in the array.
	dim3 dimBlock(threadsPerBlock, threadsPerBlock); //16x16 thread blocks
	dim3 dimGrid(blocksPerGrid, blocksPerGrid);

	//Run the selected kernel
	if (filter_type == '8'){//CUDA Box Blur
		float boxblur_stencil[9] = { 1.f / 9 , 1.f / 9, 1.f / 9 , 1.f / 9, 1.f / 9, 1.f / 9 , 1.f / 9, 1.f / 9 , 1.f / 9 };
		HANDLE_CUDA_ERROR(cudaMemcpy(device_stencil, boxblur_stencil, sizeof(float)*9, cudaMemcpyHostToDevice), __LINE__);

		convolutionKernel << <dimGrid, dimBlock >> >(device_in_arr, device_out_arr, width, height, device_stencil);
		HANDLE_CUDA_ERROR(cudaGetLastError(), __LINE__);

		HANDLE_CUDA_ERROR(cudaFree(device_stencil), __LINE__);//Why is this giving an error?
	}
	else if (filter_type == '9'){//CUDA Sobel Operator
		int const sobel_stencil[6][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 },
		{ -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
		//sobelKernel << <dimGrid, dimBlock >> >(device_in_arr, device_out_arr, width, height, sobel_stencil);
	}
	//Copy filtered array out of device and back to host
	HANDLE_CUDA_ERROR(cudaMemcpy(out_arr, device_out_arr, sizeof(int)*numElements, cudaMemcpyDeviceToHost), __LINE__);

	//Free allocated device arrays and reset the device.
	HANDLE_CUDA_ERROR(cudaFree(device_in_arr), __LINE__);
	HANDLE_CUDA_ERROR(cudaFree(device_out_arr), __LINE__);
	cudaDeviceReset();

	cout << "Finished Applying Filter." << endl << endl;
}