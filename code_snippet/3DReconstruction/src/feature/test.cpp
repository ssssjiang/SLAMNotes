#include "FeatureManager.h"

int main()
{
    string p1 = "/home/chenyu/图片/cathedral/100_7100.JPG";
    string p2 = "/home/chenyu/图片/cathedral/100_7101.JPG";

    Mat img1 = imread(p1);
	Mat img2 = imread(p2);

	imresize(img1, 480);
	imresize(img2, 480);

    
    vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
    vector<DMatch> matches_all;
    
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
    // drawMatches(img1, kp1, img2, kp2, matches_all, orbMatches);
    // #ifdef SAVE_IMAGE
    // saveImage(orbMatches, path + "allOrbMatches");
    // #endif

    // filter wrong correspondences
    vector<KeyPoint> rKeypoint1, rKeypoint2;
    vector<DMatch> fwMatches;

    FeatureManager manager;
    string type = "RANSAC";
    manager.filterWrongMatches(img1, img2, kp1, kp2, rKeypoint1, rKeypoint2, matches_all, fwMatches, type);
    cout << "after " << type << " filter, there are " << fwMatches.size() << "matches" << endl;
    drawMatches(img1, rKeypoint1, img2, rKeypoint2, fwMatches, orbMatches);
    imshow("matches", orbMatches);

    waitKey();
}