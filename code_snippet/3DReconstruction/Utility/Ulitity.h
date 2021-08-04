#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <cstdio>
using namespace std;
using namespace cv;


// save a mat of OpenCV type in local image type
// param img: image want to be saved
// param path: the local path where image to be saved 
void saveImage(Mat img, string path)
{
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	try {
		imwrite(path + ".png", img, compression_params);
	}
	catch (cv::Exception& ex) {
		cout << "Exception converting image to PNG format: %s\n" << ex.what();
		return;
	}
}


Mat downSample(Mat img)
{
    Mat smaller;
    smaller.create(img.rows / 2, img.cols / 2, img.type());
    resize(img, smaller, Size(smaller.cols, smaller.rows), 0, 0, CV_INTER_LINEAR);
    return smaller;
}

#endif