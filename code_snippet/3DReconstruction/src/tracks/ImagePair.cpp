#include "ImagePair.h"

ImagePair::ImagePair(bool im)
{
    this->isMatched = im;
}

ImagePair::ImagePair(Mat img1, Mat img2, bool im)
{
    this->img1 = img1.clone();
    this->img2 = img2.clone();
    this->isMatched = im;
}

ImagePair::ImagePair(unsigned int s, unsigned int e, bool im)
{
    this->i = s;
    this->j = e;
    this->isMatched = im;
}

ImagePair::~ImagePair()
{

}

void ImagePair::setIsMatched(bool im)
{
    this->isMatched = im;
}

bool ImagePair::getIsMatched() const
{
    return this->isMatched;
}

void ImagePair::setStart(unsigned int s)
{
    this->i = s;
}

unsigned ImagePair::getStart() const
{
    return this->i;
}

void ImagePair::setEnd(unsigned e)
{
    this->j = e;
}

unsigned ImagePair::getEnd() const
{
    return this->j;
}

void ImagePair::setImage1(Mat img)
{
    this->img1 = img.clone();
}

void ImagePair::setImage2(Mat img)
{
    this->img2 = img.clone();
}

vector<KeyPoint> ImagePair::getKeypoints1() const
{
    return this->keypoints1;
}

vector<KeyPoint> ImagePair::getKeypoints2() const
{
    return this->keypoints2;
}

Mat ImagePair::getDescriptor1() const
{
    return this->descriptor1;
}

Mat ImagePair::getDescriptor2() const
{
    return this->descriptor2;
}

void ImagePair::setMatches(vector<DMatch> m)
{
    this->trueMatches = m;
}

vector<DMatch> ImagePair::getMatches() const
{
    return this->trueMatches;
}

void ImagePair::startMatch()
{
    vector<KeyPoint> keypoints1, keypoints2;

    Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	orb->detectAndCompute(this->img1, Mat(), keypoints1, this->descriptor1);
    orb->detectAndCompute(this->img2, Mat(), keypoints2, this->descriptor2);

    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> allMatches;
    matcher.match(this->descriptor1, this->descriptor2, allMatches);

    FeatureManager featureManager;
    featureManager.filterWrongMatches(this->img1, this->img2, keypoints1, keypoints2,
    this->keypoints1, this->keypoints2, allMatches, this->trueMatches, "GMS");
}

void ImagePair::estimateFundamentalMat()
{
    FeatureManager featureManager;
    vector<KeyPoint> rkpts1, rkpts2;
    vector<DMatch> rMatches;

    featureManager.filterWrongMatches(this->img1, this->img2, this->keypoints1, this->keypoints2,
    rkpts1, rkpts2, this->trueMatches, rMatches, "RANSAC");

    this->keypoints1.clear();
    this->keypoints2.clear();
    this->trueMatches.clear();

    // updates keypoints and matches
    for(int i = 0; i < rMatches.size(); i++)
    {
        this->keypoints1.push_back(rkpts1[i]);
        this->keypoints2.push_back(rkpts2[i]);
        this->trueMatches.push_back(rMatches[i]);
    }
}

void ImagePair::estimateEssentialMat(Mat K)
{
    // according to H&Z (9.12)
    this->essentialMat = K.t() * this->fundamentalMat * K;
}

Mat ImagePair::getFundamentalMat() const
{
    return this->fundamentalMat;
}

Mat ImagePair::getEssentialMat() const
{
    return this->essentialMat;
}