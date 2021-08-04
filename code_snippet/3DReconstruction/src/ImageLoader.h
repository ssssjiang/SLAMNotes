#ifndef IMAGE_LOADER_H
#define IMAGF_LOADER_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class ImageLoader
{
private:
	vector<Mat> imgBuffer;
	string path;

private:
	void load();

public:
	void loadImage();
	vector<Mat> getImgBuffer() const;
	string getPath() const;
};

#endif