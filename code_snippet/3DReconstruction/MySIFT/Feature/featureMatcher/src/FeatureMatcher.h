#ifndef FEATUREMATCHER_H 
#define FEATUREMATCHER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace std;
using namespace cv;

class FeatureMatcher
{
private:
    vector<KeyPoint> kp1;
    vector<KeyPoint> kp2;

public:
    void getKeypoints(vector<Point2d>& imgPoints1, vector<Point2d>& imgPoints2, 
    vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<DMatch> goodMatches);
    void startMatch(Mat img1, Mat img2, vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2);
};

#endif