#ifndef UTILITY_H
#define UTILITY_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <ctime>
#include <sys/time.h> 

using namespace cv;
using namespace std;

void saveImage(Mat img, string path);

double getCurrentTime();

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

void imresize(Mat &src, int height);

#endif