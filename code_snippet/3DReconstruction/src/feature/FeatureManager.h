#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H


#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>

#include "GMSMatcher.h"

using namespace std;
using namespace cv;

class FeatureManager
{

public:
    void filterWrongMatches(Mat img1, Mat img2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
        vector<KeyPoint>& rKeypoints1, vector<KeyPoint>& rKeypoints2,
        vector<DMatch> matches, vector<DMatch>& fwMatches, string type);

private:
    void ransacFilter(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
    vector<KeyPoint>& rKeypoints, vector<KeyPoint>& rKeypoints2,
    vector<DMatch> matches, vector<DMatch>& rMatches);

    void GMSFilter(Mat img1, Mat img2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
        vector<KeyPoint>& rKeypoints1, vector<KeyPoint>& rKeypoints2,
        vector<DMatch> matches, vector<DMatch>& gmsMatches);

};

#endif