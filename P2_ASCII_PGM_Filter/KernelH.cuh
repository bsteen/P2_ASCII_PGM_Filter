#include <device_launch_parameters.h>
void launchKernel(int  * in_arr, int  * out_arr, int width, int height, char const filter_type);
void HANDLE_CUDA_ERROR(cudaError_t err, int line_num);