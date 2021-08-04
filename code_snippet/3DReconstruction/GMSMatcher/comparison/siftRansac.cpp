#include "siftRansac.h"


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

void detectSIFTKeypoints(Mat img1, Mat img2, Ptr<Feature2D> f2d, 
    vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2)
{   
    // Detect the keypoints
    f2d->detect(img1, kpts1);
    f2d->detect(img2, kpts2);
}

void computeSIFTDescriptors(Mat img1, Mat img2, Ptr<Feature2D> f2d, 
    vector<KeyPoint> kpts1, vector<KeyPoint> kpts2,
    Mat& des1, Mat& des2)
{
    f2d->compute(img1, kpts1, des1);
    f2d->compute(img2, kpts2, des2);
}

void MyRansac(Mat img1, Mat img2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, 
               Mat descriptor1, Mat descriptor2, vector<DMatch> matches, 
               vector<DMatch>& rMatches, vector<KeyPoint>& rKeypoint1, vector<KeyPoint>& rKeypoint2)
{
        // --------------using ransac filtering wrong correspondences------------
    
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
}