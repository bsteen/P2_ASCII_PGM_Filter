#include "Main.cuh"
#include "Load.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

//Load the image through the terminal with ./filter.out < image.pgm
void directLoadImage(int *image_array, int &width, int &height, int &grayscale, string &file){
	file = "file";
	string line;
	string dimensions[2];
	int i = 0;

	getline(cin, line);//Get .pgm "Magic number"
	if (line == "P2")
	{
		cout << "Valid P2/ASCII .pgm file found." << endl;
	}
	else{
		errorExit("Not a valid File.  Must be .pgm P2/ASCII.");
	}

	//Skip over any comments
	getline(cin, line);
	while (line.at(0) == '#'){
		getline(cin, line);
	}

	//Get width and height dimensions of image
	//Use string stream to break line up into a string array then convert to ints
	stringstream ssin(line);
	while (ssin && i < 2){
		ssin >> dimensions[i];
		i++;
	}

	width = stoi(dimensions[0]);//Now convert strings to integers
	height = stoi(dimensions[1]);

	if (width > 1024 || height > 1024){//Change this check to accommodate larger images. Make sure enough memory is allocated.
		errorExit("Incorrect image dimensions. Max size is 1024x1024.");
	}
	else{
		cout << "The image dimensions are : " << width << "x" << height << endl;
	}

	//Get grayscale value
	getline(cin, line);
	grayscale = stoi(line);//String to int conversion
	cout << "The grayscale range for this image is 0 to " << grayscale << "." << endl;

	//Store numbers into the 2D int array
	for (int i = 0; i < width * height; i++){
		cin >> ws;//Extracts as many whitespace characters as possible from the current position in the input sequence.
		getline(cin, line, ' ');
		image_array[i] = stoi(line);
	}

	//**MUST FIND WAY TO REDIRECT OR CLEAR CIN FROM TAKING IN THE FILE AFTER THIS POINT************************************************

	cout << "Finished loading image." << endl << endl;
}

//Load the image by typing in the file name;
void manualLoadImage(int *image_array, int &width, int &height, int &grayscale, string &file)
{
	string image_path;
	string line;
	string dimensions[2];
	int i = 0;

	cout << "Load which image from Images sub-folder? (Enter file name only) ";
	getline(cin, file);

	cout << endl << "Loading image \""+ file +".pgm\" from Images..." << endl;
	image_path ="Images/" +file + ".pgm";

	//Load .pgm (P2/ASCII Format) image into a 2D array.
	ifstream image_in(image_path);
	if (image_in)
	{
		getline(image_in, line);//Get .pgm "Magic number"
		if (line == "P2")
		{
			cout << "Valid P2/ASCII .pgm file found." << endl;
		}
		else{
			errorExit("Not a valid File.  Must be .pgm P2/ASCII.");
		}

		//Skip over any comments
		getline(image_in, line);
		while (line.at(0) == '#'){
			getline(image_in, line);
		}

		//Get width and height dimensions of image
		//Use string stream to break line up into a string array then convert to ints
		stringstream ssin(line);
		while (ssin && i < 2){
			ssin >> dimensions[i];
			i++;
		}

		width = stoi(dimensions[0]);//Now convert strings to integers
		height = stoi(dimensions[1]);

		if (width > 1024 || height > 1024){//Change this check to accommodate larger images. Make sure enough memory is allocated.
			errorExit("Incorrect image dimensions. Max size is 1024x1024.");
		}
		else{
			cout << "The image dimensions are : " << width << "x" << height << endl;
		}

		//Get grayscale value
		getline(image_in, line);
		grayscale = stoi(line);//String to int conversion
		cout << "The grayscale range for this image is 0 to " << grayscale << "." << endl;

		//Store numbers into the 2D int array
		for (int i = 0; i < width * height; i++){
			image_in >> ws;//Extracts as many whitespace characters as possible from the current position in the input sequence.
			getline(image_in, line, ' ');
			image_array[i] = stoi(line);
		}
		image_in.close();//close ifstream
	}
	else{
		errorExit("File not found.");
	}

	cout << "Finished loading image." << endl << endl;
}
