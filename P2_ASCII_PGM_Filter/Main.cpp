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

void saveImage(int * image_array, int width, int height, int grayscale){
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

void getSurroundingPixels(int width, int height, int position, int * ul, int * um, int * ur, int * ml, int * mr, int * ll, int * lm, int * lr){

	if (position % width != 0){//Check that the postion is not on the left edge of the 2D array
		*ml = position - 1;
		if (position - height > 0){
			*ul = *ml - height;
		}
		if (position + height < height*width){
			*ll = *ml + height;
		}
	}

	if (position - height > 0){///Check that the postion is not in the top row of the array.
		*um = position - height;
	}

	if (position + height < position * width){//Check that the postion is not in the top row of the array.
		*lm = position + height;
	}

	if ((position + 1) % width != 0){//Check that the postion is not on the right edge of the 2D array
		*mr = position + 1;
		if (position - height  > 0){
			*ur = *mr - height;
		}
		if (position + height < height*width){
			*lr = *mr + height;
		}
	}
}

//Broken
void serialSobelFilter(int * image_array, int width, int height){
	cout << "Applying Sobel Filter..." << endl;

	//Make array to hold new, filtered values of the image array.
	int * filtered_ia = (int *)malloc(sizeof(image_array));

	for (int r = 0; r < height; r++){
		for (int c = 0; c < width; c++){
			int mm = r*width + c;//Row Major layout
			int * ul = 0, *um = 0, *ur = 0, *ml = 0, *mr = 0, *ll = 0, *lm = 0, *lr = 0;

			getSurroundingPixels(width, height, mm, ul, um, ur, ml, mr, ll, lm, lr);

			int horz = image_array[*ur] + 2 * image_array[*mr] + image_array[*lr] - image_array[*ul] - 2 * image_array[*ml] - image_array[*ll];
			int vert = image_array[*ul] + 2 * image_array[*um] + image_array[*ur] - image_array[*ll] - 2 * image_array[*lm] - image_array[*lr];
			filtered_ia[mm] = abs(horz) + abs(vert);
		}
	}

	//Free orginal array
	free(image_array);

	//Copy filtered data back to oringal array
	memcpy(image_array, filtered_ia, sizeof(filtered_ia));
	free(filtered_ia);

	cout << "Finished Applying Sobel Filter." << endl << endl;
}

//First-chance exception at 0x00007FF6E5FD327B in MySobelFilter.exe: 0xC0000005: Access violation reading location 0x0000000000000000.
void serialBlurFilter(int * image_array, int width, int height){
	cout << "Applying Blur Filter..." << endl;

	//Make array to hold new, filtered values of the image array.
	int * filtered_ia = (int *)malloc(sizeof(image_array));

	for (int r = 0; r < height; r++){
		for (int c = 0; c < width; c++){
			int mm = r*width + c;//Row Major layout

			int *ul = 0, *um = 0, *ur = 0, *ml = 0, *mr = 0, *ll = 0, *lm = 0, *lr = 0;
			//getSurroundingPixels(width, height, mm, ul, um, ur, ml, mr, ll, lm, lr);

			filtered_ia[mm] = (*ul + *um + *ur + *ml + *mr + *ll + *lm + *lr) / 8;
		}
	}

	//Copy filtered data back to oringal array
	memcpy(image_array, filtered_ia, sizeof(filtered_ia));
	free(filtered_ia);

	cout << "Finished Applying Blur Filter." << endl << endl;
}

int main(void){
	cout << "P2/ASCII PGM (Portable Gray Map) 1024x1024  Blur & Sobel Filter" << endl << endl;

	//If you want bigger image files to be accpeted, change this allocated memory size accordingly.
	int * image_array = (int *)malloc(sizeof(int) * 1048576);//1024*1024 = 1048576
	int * width = (int *)malloc(sizeof(int));
	int * height = (int *)malloc(sizeof(int));
	int * grayscale = (int *)malloc(sizeof(int));

	loadImage(image_array, width, height, grayscale);

	//serialSobelFilter(image_array, *width, *height);
	//serialBlurFilter(image_array, *width, *height);

	saveImage(image_array, *width, *height, *grayscale);

	free(image_array);
	free(width);
	free(height);
	free(grayscale);

	cout << "Done." << endl;
	return 0;
}