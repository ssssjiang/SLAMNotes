#ifndef SIFTRANSAC_H
#define SIFTRANSAC_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
//#include "../../Utility/Utility.h"
using namespace std;
using namespace cv;

const string path = "/home/chenyu/桌面/三维重建/code/3DReconstruction/GMSMatcher/comparison/";

void saveImage(Mat img, string path);

void detectSIFTKeypoints(Mat img1, Mat img2, Ptr<Feature2D> f2d, 
    vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2);

void computeSIFTDescriptors(Mat img1, Mat img2, Ptr<Feature2D> f2d, 
    vector<KeyPoint> kpts1, vector<KeyPoint> kpts2,
    Mat& des1, Mat& des2);

void MyRansac(Mat img1, Mat img2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, 
    Mat descriptor1, Mat descriptor2, vector<DMatch> matches,
    vector<DMatch>& rMatches, vector<KeyPoint>& rKeypoint1, vector<KeyPoint>& rKeypoint2);

#endif