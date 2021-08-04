#include "ImageLoader.h"
#include "FeatureManager.h"
#include "ImagePair.h"


int main()
{
    ImageLoader* imageLoader = new ImageLoader();
    imageLoader->loadImage();

    ImagePair imgPair(false);
    Mat img1 = imageLoader->getImgBuffer()[0];
    Mat img2 = imageLoader->getImgBuffer()[1];

    imgPair.setImage1(img1);
	imgPair.setImage2(img2);

    
    vector<KeyPoint> kp1, kp2;
    vector<DMatch> matches_all;
    
    imgPair.startMatch();
    kp1 = imgPair.getKeypoints1();
    kp2 = imgPair.getKeypoints2();
    matches_all = imgPair.getMatches();

    Mat orbMatches;
    drawMatches(img1, kp1, img2, kp2, matches_all, orbMatches);
    imshow("matches", orbMatches);


    delete imageLoader;

    waitKey();
}