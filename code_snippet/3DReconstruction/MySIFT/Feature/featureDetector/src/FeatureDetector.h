#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

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

void saveImage(Mat img, string path);
Mat downSample(Mat img);
Mat** buildGaussPyramid(Mat inputImg, int octaves, int intervals, double sig);
Mat** buildDogPyramid(Mat** gp, int octaves, int intervals);
void localExtremaDetect(Mat** dogPyramid);

#endif