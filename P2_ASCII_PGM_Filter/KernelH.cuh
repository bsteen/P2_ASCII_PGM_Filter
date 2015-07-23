#include <device_launch_parameters.h>
void launchKernel(int const  * in_arr, int  * out_arr, int const width, int const height, char filter_type);
void HANDLE_CUDA_ERROR(cudaError_t err, int line_num);