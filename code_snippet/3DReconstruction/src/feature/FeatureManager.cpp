#include "FeatureManager.h"


void FeatureManager::ransacFilter(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,  
    vector<KeyPoint>& rKeypoints1, vector<KeyPoint>& rKeypoints2,
    vector<DMatch> matches, vector<DMatch>& rMatches)
    {
        // --------------using ransac filtering wrong correspondences------------
        // RANSAC消除误匹配特征点 主要分为三个部分
        // 1）根据matches将特征点对其，将坐标转换为float类型
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
                rKeypoints1.push_back(keypoints1[matches[i].queryIdx]);
                rKeypoints2.push_back(keypoints2[matches[i].trainIdx]);
                matches[i].queryIdx = index;
                matches[i].trainIdx = index;
                rMatches.push_back(matches[i]);
                index++;
            }
        }
    }


void FeatureManager::GMSFilter(Mat img1, Mat img2, 
    vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
    vector<KeyPoint>& rKeypoints1, vector<KeyPoint>& rKeypoints2,
    vector<DMatch> matches, vector<DMatch>& gmsMatches)
    {
        imresize(img1, 480);
        imresize(img2, 480);
        
        vector<DMatch> tmpMatches;
        // GMS filter
        int num_inliers = 0;
        std::vector<bool> vbInliers;

        gms_matcher gms(keypoints1, img1.size(), keypoints2, img2.size(), matches);

        num_inliers = gms.GetInlierMask(vbInliers, false, false);

        //cout << "after gms, there are " << num_inliers << " orb matches.\n" << endl;

        // draw matches
        for (size_t i = 0; i < vbInliers.size(); ++i)
        {
            if (vbInliers[i] == true)
            {
                tmpMatches.push_back(matches[i]);
            }
        }
        
        int index = 0;
        for(int i = 0; i < tmpMatches.size(); i++)
        {
            rKeypoints1.push_back(keypoints1[tmpMatches[i].queryIdx]);
            rKeypoints2.push_back(keypoints2[tmpMatches[i].trainIdx]);
            tmpMatches[i].queryIdx = index;
            tmpMatches[i].trainIdx = index;
            gmsMatches.push_back(tmpMatches[i]);
            index++;
        }
    }


    void FeatureManager::filterWrongMatches(Mat img1, Mat img2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
        vector<KeyPoint>& fwKeypoints1, vector<KeyPoint>& fwKeypoints2,
        vector<DMatch> matches, vector<DMatch>& fwMatches, string type)
        {
            if(type == "RANSAC")
            {
                ransacFilter(keypoints1, keypoints2, fwKeypoints1, fwKeypoints2, matches, fwMatches);
            }

            else if(type == "GMS")
            {
                GMSFilter(img1, img2, keypoints1, keypoints2, fwKeypoints1, fwKeypoints2, matches, fwMatches);
            }

            else
            {
                cout << "Error occured in FeatureManager: ";
                cout << "match type[ " << type << " ] doesn't exist!" << endl; 
            }
        }


