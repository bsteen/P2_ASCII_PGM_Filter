//CUDA Headers
#include <cuda.h>
#include <cuda_runtime.h>
//C++ Headers
#include <iostream>
using namespace std;

__global__ void sobelKernel(){

}

void prepareKernel(int const  * in_arr, int  * out_arr, int const width, int const height, char filter_type){

	//Check to see if there is a CUDA enabled device
	int *count=0;
	cudaError err= cudaGetDeviceCount(count);
	if (err == cudaErrorNoDevice){
		cout << "No CUDA enabled device found on this machince."<<endl;
		exit(EXIT_FAILURE);
	}
	else if (err == cudaErrorInsufficientDriver){//These warnings were taken straight from the CUDA documentation.
		cout << "Your installed NVIDIA CUDA driver is older than the CUDA runtime library currently in use." << endl;
		cout << "Please install an updated NVIDIA display driver to allow the program to run."<< endl;
		cudaRuntimeGetVersion(count);
		cout << "The runtime version that this program uses: " + *count << endl;
		exit(EXIT_FAILURE);
	}
	cout << "CUDA enabled device found." << endl;
	cout << "Applying Filter..." << endl;

	int * cuda_in_arr;
	int * cuda_out_arr;
	cudaMalloc(&cuda_in_arr, sizeof(int)*width*height);
	cudaMalloc(&cuda_out_arr, sizeof(int)*width*height);

	if (filter_type=='8'){//Cuda Sobel Operator

	}
	
	cudaFree(cuda_in_arr);
	cudaFree(cuda_out_arr);

	cout << "Finished Applying Filter." << endl << endl;
}