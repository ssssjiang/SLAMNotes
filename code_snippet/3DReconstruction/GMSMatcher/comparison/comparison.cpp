#include "siftRansac.h"

// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 

#include "Header.h"
#include "gms_matcher.h"
#include <string>
#include <ctime>
#include <sys/time.h> 
using namespace std;

#define SAVE_IMAGE 0

double get_wall_time()  
{  
    struct timeval time ;  
    if (gettimeofday(&time,NULL)){  
        return 0;  
    }  
    return (double)time.tv_sec + (double)time.tv_usec * .000001;  
}  


void GmsMatchWithORB(Mat& img1, Mat& img2)
{
    vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
    vector<DMatch> matches_all, matches_gms;
    
    // using orb
	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
    orb->detectAndCompute(img2, Mat(), kp2, d2);

    BFMatcher matcher(NORM_HAMMING);
    matcher.match(d1, d2, matches_all);
    cout << "before ransac and gms, there are " << matches_all.size() << " orb matches.\n";
    
    // all sift matches
    Mat orbMatches;
    drawMatches(img1, kp1, img2, kp2, matches_all, orbMatches);
    #ifdef SAVE_IMAGE
    saveImage(orbMatches, path + "allOrbMatches");
    #endif

    // orb + ransac to filter wrong correspondences
    vector<KeyPoint> rKeypoint1, rKeypoint2;
    vector<DMatch> ranMatches;

    double rst = get_wall_time();
    MyRansac(img1, img2, kp1, kp2, d1, d2, matches_all, ranMatches, rKeypoint1, rKeypoint2);
    double ret = get_wall_time();
    cout << "time using ransac: " << ret - rst << " ms" << endl;

    cout << "after ransac, there are " << ranMatches.size() << " orb matches.\n";

    Mat orbransacMatches;
    drawMatches(img1, rKeypoint1, img2, rKeypoint2, ranMatches, orbransacMatches);
    #ifdef SAVE_IMAGE
    saveImage(orbransacMatches, path + "orb+ransac");
    #endif

	// GMS filter
	int num_inliers = 0;
    std::vector<bool> vbInliers;

    double gt1 = get_wall_time();
    gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
    double gt2 = get_wall_time();
    cout << "time using gms: " << gt2 - gt1 << " ms" << endl;

	num_inliers = gms.GetInlierMask(vbInliers, false, false);

	cout << "after gms, there are " << num_inliers << " orb matches.\n" << endl;

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}


    vector<KeyPoint> rkp1, rkp2;
    vector<DMatch> rMatches;
    int index = 0;
    for(int i = 0; i < matches_gms.size(); i++)
    {
        rkp1.push_back(kp1[matches_gms[i].queryIdx]);
        rkp2.push_back(kp2[matches_gms[i].trainIdx]);
        matches_gms[i].queryIdx = index;
        matches_gms[i].trainIdx = index;
        rMatches.push_back(matches_gms[i]);
        index++;
    }
    Mat show;
    drawMatches(img1, rkp1, img2, rkp2, rMatches, show);
    #ifdef SAVE_IMAGE
    saveImage(show, path + "orb+gms");
    #endif
}

void GmsMatch(Mat &img1, Mat &img2){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;
    
    // using sift
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    detectSIFTKeypoints(img1, img2, sift, kp1, kp2);
    computeSIFTDescriptors(img1, img2, sift, kp1, kp2, d1, d2);

    BFMatcher matcher;
    matcher.match(d1, d2, matches_all);
    cout << "before ransac and gms, there are " << matches_all.size() << " sift matches.\n";

    // all sift matches
    Mat siftMatches;
    drawMatches(img1, kp1, img2, kp2, matches_all, siftMatches);
    #ifdef SAVE_IMAGE
    saveImage(siftMatches, path + "allSiftMatches");
    #endif

    // sift + ransac to filter wrong correspondences
    vector<KeyPoint> rKeypoint1, rKeypoint2;
    vector<DMatch> ranMatches;

    double rst = get_wall_time();
    MyRansac(img1, img2, kp1, kp2, d1, d2, matches_all, ranMatches, rKeypoint1, rKeypoint2);
    double ret = get_wall_time();
    cout << "time using ransac: " << ret - rst << " ms" << endl;

    cout << "after ransac, there are " << ranMatches.size() << " sift matches.\n";

    Mat siftransacMatches;
    drawMatches(img1, rKeypoint1, img2, rKeypoint2, ranMatches, siftransacMatches);
    saveImage(siftransacMatches, path + "sift+ransac");

	// GMS filter
	int num_inliers = 0;
    std::vector<bool> vbInliers;
    
    double gst = get_wall_time();
    gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
    double gEt = get_wall_time();
    cout << "time using gms: " << gEt - gst << " ms" << endl;

	num_inliers = gms.GetInlierMask(vbInliers, false, false);

	cout << "after gms, there are " << num_inliers << " sift matches.\n" << endl;

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}


    vector<KeyPoint> rkp1, rkp2;
    vector<DMatch> rMatches;
    int index = 0;
    for(int i = 0; i < matches_gms.size(); i++)
    {
        rkp1.push_back(kp1[matches_gms[i].queryIdx]);
        rkp2.push_back(kp2[matches_gms[i].trainIdx]);
        matches_gms[i].queryIdx = index;
        matches_gms[i].trainIdx = index;
        rMatches.push_back(matches_gms[i]);
        index++;
    }
    Mat show;
    drawMatches(img1, rkp1, img2, rkp2, rMatches, show);
    #ifdef SAVE_IMAGE
    saveImage(show, path + "sift+gms");
    #endif
}

void runImagePair(string path1, string path2){
	Mat img1 = imread(path1);
	Mat img2 = imread(path2);

	imresize(img1, 480);
	imresize(img2, 480);

    GmsMatch(img1, img2);
    GmsMatchWithORB(img1, img2);
}


int main()
{
    // string p1 = "/home/chenyu/图片/david/B00.jpg";
    // string p2 = "/home/chenyu/图片/david/B01.jpg";

    string p1 = "/home/chenyu/图片/cathedral/100_7100.JPG";
    string p2 = "/home/chenyu/图片/cathedral/100_7101.JPG";

    // string p1 = "/home/chenyu/图片/elephant/DSC07775.JPG";
    // string p2 = "/home/chenyu/图片/elephant/DSC07776.JPG";
    runImagePair(p1, p2);
    
    waitKey();
}


