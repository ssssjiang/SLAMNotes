#include "FeatureMatcher.h"
#include <vector>
using namespace std;

int main()
{
    Mat img1 = imread("/home/chenyu/图片/david/B00.jpg");
    Mat img2 = imread("/home/chenyu/图片/david/B01.jpg");

    vector<KeyPoint> kpts1, kpts2;

    FeatureMatcher matcher;
    matcher.startMatch(img1, img2, kpts1, kpts2);

    waitKey(0);
}