//Header Files
#include <stdlib.h>
#include <stdio.h>

//C++ Files
#include <iostream>
#include <fstream>//file stream
#include <sstream>//string stream
using namespace std;

void loadImage(int * image_array, int * w, int * h, int * g)
{
	char ch;
	string image_name;
	string line;
	int width;
	int height;
	int grayscale;
	string dimensions[2];
	int i = 0;

	cout << "Load which image from Images subfolder? (l=lenap2.pgm, p=pepperp2.pgm)" << endl;
	ch = getchar();

	switch (ch){
	case 'l':
		cout << endl << "Loading image lenap2.pgm from Images..." << endl;
		image_name = "Images/lenap2.pgm";
		break;
	case 'p':
		cout << endl << "Loading image pepperp2.pgm from Images..." << endl;
		image_name = "Images/pepperp2.pgm";
		break;
	default:
		cout << endl << "ERROR: File not found." << endl;
		exit(EXIT_FAILURE);
	}

	//Load .pgm (P2/ASCII Format) Image into a 2D short array.
	ifstream image_in(image_name);
	if (image_in)
	{
		getline(image_in, line);//Get .pgm "Magic number"
		if (line == "P2")
		{
			cout << "Valid P2/ASCII .pgm file found." << endl;
		}
		else{
			cout << "Not valid File.  Must be .pgm P2/ASCII." << endl;
			exit(EXIT_FAILURE);
		}

		//Skip over any comments
		getline(image_in, line);
		while (line.at(0) == '#'){
			getline(image_in, line);
		}

		//Get width and height dimensions of image
		//Right now, line contains: "width  height" (two spaces in between them)
		//Use string stream to break it up into a string array then convert to ints
		stringstream ssin(line);
		while (ssin && i < 2){
			ssin >> dimensions[i];
			++i;
		}
		width = atoi(dimensions[0].c_str());//Now convert to integers
		height = atoi(dimensions[1].c_str());
		if (width>1024 || height>1024){//Change this check to accommodate larger images. Make sure enough memory is allocated.
			cout << "Incorrect image dimensions. Max size 1024x1024.";
			exit(EXIT_FAILURE);
		}
		else{
			*w = width;
			*h = height;
			printf("The image dimensions are: %ix%i.\n", width, height);
		}

		//Get grayscale value
		getline(image_in, line);
		grayscale = atoi(line.c_str());
		*g = grayscale;
		printf("The grayscale range for this image is 0 to %i.\n", grayscale);

		//Store numbers into the 2D int array
		int i = 0;
		while (i<width*height){
			image_in >> std::ws;//Extracts as many whitespace characters as possible from the current position in the input sequence.
			getline(image_in, line, ' ');
			image_array[i] = atoi(line.c_str());
			i++;
		}
		image_in.close();//close ifstream
	}
	else{
		cout << "ERROR: File not found." << endl;
		exit(EXIT_FAILURE);
	}

	cout << "Finished loading image." << endl << endl;
}

void saveImage(int * image_array, const int width, const int height, const int  grayscale){
	cout << "Saving Image..." << endl;
	fstream image_out;
	image_out.open("Images/output.pgm", fstream::out);

	image_out << "P2" << endl;
	image_out << "#P2/ASCII PGM (Portable Gray Map) Sobel Filter Output Image" << endl;
	image_out << to_string(width) + "  " + to_string(height) << endl;
	image_out << to_string(grayscale) << endl;

	int total = width*height;
	int num_count = 0;
	for (int i = 0; i < total; i++){
		image_out << to_string(image_array[i]) + "  ";
		num_count++;
		if (num_count >= 12){
			image_out << endl;
			num_count = 0;
		}
	}

	image_out.close();

	cout << "Image Saved." << endl << endl;
}

int* add1pxBorder(int const * in_arr, int const width, int const height){

	int W = width + 2, H = height + 2;
	int * out =  (int*) malloc(sizeof(int)*W * H);

	memset(&out[0], 0, sizeof(int)*W);
	memset(&out[W * height], 0, sizeof(int)*W);

	for (int y = 0; y < height; y++){
		out[(y+1)*W] = out[(y+1)*W + width] = 0;
		memcpy(&out[(y + 1)*W + 1], &in_arr[y*width], sizeof(int)*width);
	}
	return out;
}


void apply2dStencil3x3(int const  * in_arr, int  * out_arr, int const width, int const height, float const stencil[3][3]){
	cout << "Applying Filter..." << endl;
	
	for (int y = 1; y < height - 1; y++){
		for (int x = 1; x < width - 1; x++){

			int p = y * width + x;

			float ul = (float)(in_arr[p - width - 1]) * stencil[0][0];
			float um = (float)(in_arr[p - width]) * stencil[0][1];
			float ur = (float)(in_arr[p - width + 1]) * stencil[0][2];

			float ml = (float)(in_arr[p - 1]) * stencil[1][0];
			float mm = (float)(in_arr[p]) *stencil[1][1];
			float mr = (float)(in_arr[p + 1]) * stencil[1][2];

			float ll = (float)(in_arr[p + width - 1]) * stencil[2][0];
			float lm = (float)(in_arr[p + width]) * stencil[2][1];
			float lr = (float)(in_arr[p + width + 1]) * stencil[2][2];

			out_arr[p] = (int) (ul + um + ur + ml + mm + mr + ll + lm + lr);
		}
	}
}

int main(void){
	cout << "P2/ASCII PGM (Portable Gray Map) 1024x1024  Blur & Sobel Filter" << endl << endl;

	//If you want bigger image files to be accpeted, change this allocated memory size accordingly.
	int * image_array = (int *)malloc(sizeof(int) * 1048576);//1024*1024 = 1048576
	int height, width, grayscale;

	loadImage(image_array, &width, &height, &grayscale);

	int * expanded = add1pxBorder(image_array, width, height);
	int * outimg = (int*)malloc(sizeof(int) * (height + 2) * (width * 2));

	const float blur_stencil[3][3] = { { 1. / 9, 1. / 9, 1. / 9 }, { 1. / 9, 1. / 9, 1. / 9 }, { 1. / 9, 1. / 9, 1. / 9 } };

	apply2dStencil3x3(expanded, outimg, width + 2, height + 2, blur_stencil);
	saveImage(outimg, width+2, height+2, grayscale);

	free(image_array);
	free(expanded);
	free(outimg);

	cout << "Done." << endl;
	return 0;
}