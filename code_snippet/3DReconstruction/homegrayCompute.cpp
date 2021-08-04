#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace std;
using namespace cv;


void getHomographyMatrix(Mat)

int main()
{
    Mat img1 = imread("/home/chenyu/桌面/三维重建/data/david/B00.jpg");
    Mat img2 = imread("/home/chenyu/桌面/三维重建/data/david/B01.jpg");

    if(img1.empty())
    {
        cout << "img1 cannot be loaded!" << endl;
        return -1;
    }

    if(img2.empty())
    {
        cout << "img2 cannot be loaded!" << endl;
        return -1;
    }

    // feature detect
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    vector<KeyPoint> kp1, kp2;

    orb->detect(img1, kp1);
    Mat outImg1;
    drawKeypoints(img1, kp1, outImg1);

    orb->detect(img2, kp2);
    Mat outImg2;
    drawKeypoints(img2, kp2, outImg2);

    imshow("img1 keypoints", outImg1);
    imshow("img2 keypoints", outImg2);

    // descriptor compute
    Mat descriptor1, descriptor2;
    orb->compute(img1, kp1, descriptor1);
    orb->compute(img2, kp2, descriptor2);

    // feature match
    vector<DMatch> matches;
    BFMatcher matcher( NORM_HAMMING);
    matcher.match(descriptor1, descriptor2, matches);

    double minDist = 10000, maxDist = 0;
    for(int i = 0; i < descriptor1.rows; i++)
    {
        double dist = matches[i].distance;
        minDist = min(minDist, dist);
        maxDist = max(maxDist, dist);
    }

    // 筛选好的匹配点
    vector<DMatch> goodMatches;
    for(int i = 0; i < descriptor1. rows; i++)
    {
        if(matches[i].distance <= max(2 * minDist, 30.0) ) goodMatches.push_back(matches[i]);
    }
    
    // draw match points
    Mat matchPoints;
    drawMatches(img1, kp1, img2, kp2, matches, matchPoints);
    imshow("match points", matchPoints);

    waitKey(0);
}