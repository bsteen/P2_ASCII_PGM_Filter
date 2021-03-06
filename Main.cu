#include "Main.cuh"
#include "Kernel.cuh"
#include "Load.cuh"
#include "Window.hpp"
#include <ctime>
#include <fstream>
#include <iostream>
using namespace std;

//P2/ASCII PGM Image Filter, Version 1.1
//This program was made by Benjamin Steenkamer.

void errorExit(string error){
	cout << error << endl << "Press ENTER to exit.";
	getchar();
	exit(EXIT_FAILURE);
}

char getFilterType(){
	cout << "Which filter would you like to run?" << endl;
	cout << "1: Box Blur (Serial)" << endl;
	cout << "2: Gaussian Approx. (Serial)" << endl;
	cout << "3: Sharpen (Serial)" << endl;
	cout << "4: Salt and Pepper (Serial)" << endl;
	cout << "5: Emboss (Serial)" << endl;
	cout << "6: Edge Detection (Serial)" << endl;
	cout << "7: Sobel Operator (Serial)" << endl;
	cout << "8: Box  Blur (CUDA)" << endl;
	cout << "9: Sobel Operator (CUDA)" << endl;

	cin >> ws; //Eat up the previous white spaces in buffer
	char type = getchar();
	cout << endl;

	return type;
}

int getFilterPasses(){
	cout << "How many times would you like to run the filter?" << endl;
	cin >> ws;

	string temp_input;
	getline(cin, temp_input);
	int passes = stoi(temp_input);
	cout << endl;

	return passes;
}

void saveImage(int* image_array, const int width, const int height, const int  grayscale, string file_name){
	cout << "Saving image as "+ file_name +"_output.pgm to the Output sub-folder..." << endl;
	fstream image_out;
	image_out.open("Output/"+ file_name + "_output.pgm", fstream::out);

	image_out << "P2" << endl;
	image_out << "#P2/ASCII PGM (Portable Gray Map) Filter Output" << endl;

	image_out << to_string(width) + " " + to_string(height) << endl;
	image_out << to_string(grayscale) << endl;

	int total = (width * height);
	int num_count = 0;

	for (int i = 0; i < total; i++){
		if (image_array[i] < 0){//This is a cheaty way of removing the corrput borders. It is not a subsitute for correct edge detection in the filter.
			image_out << to_string(0) + " ";
		}
		else{
			image_out << to_string(image_array[i]) + " ";
		}
		num_count++;

		if (num_count >= 16){//This is just to make the ASCII lines in the image file look neat/compact.
			image_out << endl;
			num_count = 0;
		}
	}

	image_out.close();

	cout << "Image saved." << endl << endl;
}

int* add1pxBorder(int* const in_arr, int const width, int const height){

	int W = width + 2;
	int H = height + 2;

	int* out = (int*)malloc(sizeof(int)* W * H);

	memset(&out[0], 0, sizeof(int) * W);
	memset(&out[W * height], 0, sizeof(int) * W);

	for (int y = 0; y < height; y++){
		out[(y + 1)*W] = out[(y + 1)*W + width] = 0;
		memcpy(&out[(y + 1) * W + 1], &in_arr[y * width], sizeof(int) * width);
	}

	return out;
}

int* remove1pxBorder(int* const in_arr, int const width, int const height){
	int *out_arr = (int*)malloc(sizeof(int) * width * height);

	for (int y = 0; y < height; y++){
		memcpy(&out_arr[y * width], &in_arr[(y + 1) * (width+2)+1], sizeof(int) * width);
	}

	return out_arr;
}

void applyConvolutionStencil(int* const in_arr, int* out_arr, int p, int const width, int const height, const float stencil[3][3]){

	float ul = (float)(in_arr[p - width - 1]) * stencil[0][0];
	float um = (float)(in_arr[p - width]) * stencil[0][1];
	float ur = (float)(in_arr[p - width + 1]) * stencil[0][2];

	float ml = (float)(in_arr[p - 1]) * stencil[1][0];
	float mm = (float)(in_arr[p]) *stencil[1][1];
	float mr = (float)(in_arr[p + 1]) * stencil[1][2];

	float ll = (float)(in_arr[p + width - 1]) * stencil[2][0];
	float lm = (float)(in_arr[p + width]) * stencil[2][1];
	float lr = (float)(in_arr[p + width + 1]) * stencil[2][2];

	out_arr[p] = (int)(ul + um + ur + ml + mm + mr + ll + lm + lr);
}

void applySobelStencil(int* const in_arr, int* out_arr, int p, int const width, int const height, const int stencil[6][3]){

	int ul = (in_arr[p - width - 1]) * stencil[0][0];
	int um = (in_arr[p - width]) * stencil[0][1];
	int ur = (in_arr[p - width + 1]) * stencil[0][2];
	int ml = (in_arr[p - 1]) * stencil[1][0];
	int mm = (in_arr[p]) *stencil[1][1];
	int mr = (in_arr[p + 1]) * stencil[1][2];
	int ll = (in_arr[p + width - 1]) * stencil[2][0];
	int lm = (in_arr[p + width]) * stencil[2][1];
	int lr = (in_arr[p + width + 1]) * stencil[2][2];
	int x_sum = ul + um + ur + ml + mm + mr + ll + lm + lr;

	ul = (in_arr[p - width - 1]) * stencil[3][0];
	um = (in_arr[p - width]) * stencil[3][1];
	ur = (in_arr[p - width + 1]) * stencil[3][2];
	ml = (in_arr[p - 1]) * stencil[4][0];
	mm = (in_arr[p]) *stencil[4][1];
	mr = (in_arr[p + 1]) * stencil[4][2];
	ll = (in_arr[p + width - 1]) * stencil[5][0];
	lm = (in_arr[p + width]) * stencil[5][1];
	lr = (in_arr[p + width + 1]) * stencil[5][2];
	int y_sum = ul + um + ur + ml + mm + mr + ll + lm + lr;

	out_arr[p] = (int)pow((y_sum * y_sum + x_sum * x_sum), 0.5);
}

void runFilter(int* const in_arr, int* out_arr, int const width, int const height, char filter_type){
		for (int y = 1; y < height - 1; y++){
			for (int x = 1; x < width - 1; x++){
				int p = y * width + x;
				if (filter_type == '1'){//Box Blur
					float const boxblur_stencil[3][3] = { { 1.f / 9, 1.f / 9, 1.f / 9 }, { 1.f / 9, 1.f / 9, 1.f / 9 }, { 1.f / 9, 1.f / 9, 1.f / 9 } };
					applyConvolutionStencil(in_arr, out_arr, p, width, height, boxblur_stencil);
				}
				else if (filter_type == '2'){//Gaussian blur
					float const gauss_stencil[3][3] = { { 1.f / 16, 1.f / 8, 1.f / 16 }, { 1.f / 8, 1.f / 4, 1.f / 8 }, { 1.f / 16, 1.f / 8, 1.f / 16 } };
					applyConvolutionStencil(in_arr, out_arr, p, width, height, gauss_stencil);
				}
				else if (filter_type == '3'){//Sharpen
					float const sharpen_stencil[3][3] = { { 0, -1, 0 }, { -1, 5, -1 }, { 0, -1, 0 } };
					applyConvolutionStencil(in_arr, out_arr, p, width, height, sharpen_stencil);
				}
				else if (filter_type == '4'){//Salt and Pepper
					float const sp_stencil[3][3] = { { 0, 0, 0 }, { -1, 1, 0 }, { 0, 0, 0 } };
					applyConvolutionStencil(in_arr, out_arr, p, width, height, sp_stencil);
				}
				else if (filter_type == '5'){//Emboss
					float const emboss_stencil[3][3] = { { -2, -1, 0 }, { -1, 1, 1 }, { 0, 1, 2 } };
					applyConvolutionStencil(in_arr, out_arr, p, width, height, emboss_stencil);
				}
				else if (filter_type == '6'){//Edge Detection
					float const egde_e_stencil[3][3] = { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };
					applyConvolutionStencil(in_arr, out_arr, p, width, height, egde_e_stencil);
				}
				else if (filter_type == '7'){//Sobel Operator
					int const sobel_stencil[6][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 },
					{ -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
					applySobelStencil(in_arr, out_arr, p, width, height, sobel_stencil);
				}
			}
		}
}

int main(int argc, char** argv){
	cout << "P2/ASCII PGM 1024x1024 Filter" << endl << endl;

	int height, width, grayscale;


	if(argc != 2){
		cout << "Must have one PGM image file as an argument." << endl;
		exit(EXIT_FAILURE);
	}

	string file_name(argv[1]);

	//If you want bigger image files to be accpeted, change this allocated memory size accordingly.
	//Keep in mind that you must have enough space for the additonal 1 pixel border that will go around the image during convolution.
	int* image_array = (int*)malloc(sizeof(int) * 1048576);//1024 * 1024 = 1048576

	loadImage(image_array, width, height, grayscale, file_name);

	int* expanded = add1pxBorder(image_array, width, height);
	int* outimg = (int*)malloc(sizeof(int) * (height + 2) * (width + 2));

	//Get the filter type and number of passes.
	char filter_type = getFilterType();
	int filter_passes = getFilterPasses();

	double timer = 0;
	if(filter_type < '8'){//For serial filters
		cout << "Applying Filter..." << endl;

		clock_t start = clock();//For time calculation. Start timer
		for (int c = 0; c < filter_passes; c++){
			runFilter(expanded, outimg, width + 2, height + 2, filter_type);
			memcpy(expanded, outimg, sizeof(int) * (width + 2) * (height + 2));
		}
		timer = (clock() - start) / (double)CLOCKS_PER_SEC;//Stop timer
	}
	else{//For CUDA Filters
		launchKernel(expanded, outimg, width + 2, height + 2, filter_type, filter_passes, timer);
	}

	//Print out time it took to complete filter.
	cout << "Finished Applying Filter." << endl << "It took " << timer << "s to complete. ";
	if (timer == 0){// A value 0 from the timer means the calculation time is less than 0.001s.
		cout << "(Less .001 seconds)";
	}
	cout << endl << endl;

	//Remove the border and save the image
	int* final_image = remove1pxBorder(outimg, width, height);

	cout << "Displaying final image...(Press Escape to continue)" << endl;
	displayImage(final_image, width, height);

	saveImage(final_image, width, height, grayscale, file_name);

	free(image_array);
	free(expanded);
	free(outimg);

	return 0;
}
