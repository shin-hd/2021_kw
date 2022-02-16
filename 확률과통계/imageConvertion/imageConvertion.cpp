#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
using namespace std;

// read image file data of file_name
int getFileData(string file_name, unsigned char* data, int dataSize);
// write image file data of new + file_name + .raw
void storeFileData(string file_name, unsigned char* data, int dataSize);

// get CDF of image data
void getCDF(double* CDF, unsigned char* data, int dataSize);
// write CDF of file_name.txt
void writeCDF(string file_name, double* CDF);

// create new image file data matched to the target
void imageConversion(unsigned char* source_image, double* source_cdf, int source_size, double* target_cdf);

// inverse CDF // CDF(char) = p
char inverseCDF(double* CDF, double p);

int main()
{
	string source_name = "Couple(512x512)",		// file name
		target_name = "gBaboon256_256",
		file_extension = ".raw";
	unsigned char* source_image, * target_image;// image binary
	int source_size = 512 * 512,
		target_size = 256 * 256;				// size
	double source_cdf[256], target_cdf[256];	// cdf

	// target setting //
	target_image = new unsigned char[target_size];
	if (getFileData(target_name + file_extension, target_image, target_size))
	{	// file does not exist
		cout << source_name << " : no such file" << endl;
		return -1;
	}
	getCDF(target_cdf, target_image, target_size);
	delete[] target_image;	// target data is no longer used
	// end of setting //

	// source setting //
	source_image = new unsigned char[source_size];
	if (getFileData(source_name + file_extension, source_image, source_size))
	{	// file does not exist
		cout << source_name << " : no such file" << endl;
		return -1;
	}
	getCDF(source_cdf, source_image, source_size);
	// end of setting //

	// original CDF
	writeCDF("sourceCDF", source_cdf);
	writeCDF("targetCDF", target_cdf);

	// convert and store
	imageConversion(source_image, source_cdf, source_size, target_cdf);
	storeFileData(source_name, source_image, source_size);
	
	// converted CDF
	getCDF(source_cdf, source_image, source_size);
	writeCDF("newCDF", source_cdf);
}

int getFileData(string file_name, unsigned char* data, int dataSize)
{
	ifstream read;
	read.open(file_name, ifstream::binary);
	if (!read.is_open())
	{	// file does not exist
		read.close();
		return 1;
	}

	// read
	read.read((char*)data, dataSize);
	read.close();
	return 0;
}
void storeFileData(string file_name, unsigned char* data, int dataSize)
{
	ofstream write;
	write.open("new" + file_name + ".raw");
	
	// write
	write.write((const char*)data, dataSize);
	write.close();
}

void getCDF(double* CDF, unsigned char* data, int dataSize)
{
	// initialize CDF array
	for (int i = 0; i < 256; i++)
	{
		CDF[i] = 0;
	}
	
	// get frequency of each 0~255
	for (int i = 0; i < dataSize; i++)
	{
		CDF[data[i]]++;
	}

	// get CDF
	CDF[0] = CDF[0] / dataSize;
	for (int i = 1; i < 256; i++)
	{
		// F(i) = F(i - 1) + P(i) 
		CDF[i] = (CDF[i] / dataSize) + CDF[i - 1];
	}
}

void writeCDF(string file_name, double* CDF)
{
	ofstream cdf;
	cdf.open(file_name + ".txt");
	for (int i = 0; i < 256; i++)
	{	// write
		cdf << CDF[i] << endl;
	}
	cdf.close();
}

void imageConversion(unsigned char* source_image, double* source_cdf, int source_size, double* target_cdf)
{
	for (int i = 0; i < source_size; i++)
	{	// new image = inverseTargetCDF( sourceCDF ( source image 8-bit ) )
		source_image[i] = inverseCDF(target_cdf, source_cdf[source_image[i]]);
	}
}

char inverseCDF(double* CDF, double p)
{
	for (int i = 0; i < 256; i++)
	{
		if (CDF[i] >= p)
			return (char)i;
	}
	return 255;
}