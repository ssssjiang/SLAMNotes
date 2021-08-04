#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
using namespace std;
using namespace cv;

int main()
{
    Mat img1 = imread("/home/chenyu/图片/david/B00.jpg");
    Mat img2 = imread("/home/chenyu/图片/david/B01.jpg");

    if(img1.empty() || img2.empty())
    {
        cout << "cannot load image" << endl;
        return -1;
    }

    // sift特征检测
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

    // Detect the keypoints
    vector<KeyPoint> keypoints1, keypoints2;
    f2d->detect(img1, keypoints1);
    f2d->detect(img2, keypoints2);

    // Calculate descriptors
    Mat descriptor1, descriptor2;
    f2d->compute(img1, keypoints1, descriptor1);
    f2d->compute(img2, keypoints2, descriptor2);

    // matching descriptor vector using BFMatcher
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);

    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("match", img_matches);


    // RANSAC消除误匹配特征点 主要分为三个部分
    // 1）根据matches将tezhengidan对其，将坐标转换为float类型
    // 2）使用求基础矩阵方法 findFundamentalMat， 得到RansacStatus
    // 3) 根据RansacStatus来将误匹配的点也即RansacStatus[i] = 0 的点筛除

    // 根据matches将特征点对齐，将坐标转换为float类型
    vector<Point2f> imgkpts1, imgkpts2;
    for(int i = 0; i < matches.size(); i++)
    {
        imgkpts1.push_back(keypoints1[matches[i].queryIdx].pt);
        imgkpts2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // 利用基本矩阵筛除误匹配点
    vector<uchar> RansacStatus;
    Mat fundamental = findFundamentalMat(imgkpts1, imgkpts2, RansacStatus, FM_RANSAC);

    vector<KeyPoint> rKeypoint1, rKeypoint2;
    vector<DMatch> rMatches;
    int index = 0;
    for(int i = 0; i < matches.size(); i++)
    {
        if(RansacStatus[i] != 0)
        {
            rKeypoint1.push_back(keypoints1[matches[i].queryIdx]);
            rKeypoint2.push_back(keypoints2[matches[i].trainIdx]);
            matches[i].queryIdx = index;
            matches[i].trainIdx = index;
            rMatches.push_back(matches[i]);
            index++;
        }
    }

    Mat img_r_matches;
    drawMatches(img1, rKeypoint1, img2, rKeypoint2, rMatches, img_r_matches);
    imshow("after ransac", img_r_matches);


    waitKey(0);
}