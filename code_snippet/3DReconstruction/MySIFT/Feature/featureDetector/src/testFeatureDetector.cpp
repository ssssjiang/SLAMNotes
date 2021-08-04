#include "FeatureDetector.h"


int main()
{
    Mat img = imread("/home/chenyu/图片/david/B00.jpg");
    
    Mat** gauPyramid = buildGaussPyramid(img, 3, 5, 1.6);
    
    Mat** dogPyramid = buildDogPyramid(gauPyramid, 3, 5);

    for(int o = 0; o < 3; o++)
    {   
        for(int i = 0; i < 7; i++)
        {
            //imshow(to_string(o) + "," + to_string(i), dogPyramid[o][i]);
            string root = "/home/chenyu/桌面/三维重建/code/3DReconstruction/MySIFT/Feature/featureDetector/result/";
            saveImage(dogPyramid[o][i], root + to_string(o) + "," + to_string(i));
        }
    }


    for(int o = 0; o < 3; o++)
    {   
        for(int i = 0; i < 8; i++)
        {
            string root = "/home/chenyu/桌面/三维重建/code/3DReconstruction/MySIFT/Feature/featureDetector/result/gaussian/";
            saveImage(gauPyramid[o][i], root + to_string(o) + "," + to_string(i));
        }
    }

    waitKey(0);
}