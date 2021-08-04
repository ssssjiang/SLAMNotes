#ifndef IMAGE_PAIR_H
#define IMAGE_PAIR_H

#include "../ImageLoader.h"
#include "../FeatureManager.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>


class ImagePair
{
private:
    unsigned int i;
    unsigned int j;

    bool isMatched;

    Mat img1;
    Mat img2;

    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;

    Mat descriptor1;
    Mat descriptor2;

    vector<DMatch> trueMatches;

    Mat fundamentalMat;
    Mat essentialMat;

public:
    ImagePair(bool im = false);
    ImagePair(Mat img1, Mat img2, bool im = false);
    ImagePair(unsigned int s, unsigned int e, bool im = false);
    ~ImagePair();

    void setIsMatched(bool);
    bool getIsMatched() const;

    void setStart(unsigned);
    unsigned int getStart() const;

    void setEnd(unsigned);
    unsigned int getEnd() const;

    void setImage1(Mat img);
    void setImage2(Mat img);

    vector<KeyPoint> getKeypoints1() const;
    vector<KeyPoint> getKeypoints2() const;

    Mat getDescriptor1() const;
    Mat getDescriptor2() const;

    void setMatches(vector<DMatch>);
    vector<DMatch> getMatches() const;

    void startMatch();

    void estimateFundamentalMat();
    void estimateEssentialMat(Mat K);

    Mat getFundamentalMat() const;
    Mat getEssentialMat() const;

};

#endif