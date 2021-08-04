//#include "ImageLoader.h"

#include <ctime>
#include <sys/time.h>
#include <fstream>
#include <string>

#include "Utility.h"
#include "FeatureManager.h"
#include "EGGraph.h"
#include "TrackManager.h"
#include "ImageLoader.h"

using namespace std;

double getCurrentTime()  
{  
    struct timeval time ;  
    if (gettimeofday(&time,NULL)){  
        return 0;  
    }  
    return (double)time.tv_sec + (double)time.tv_usec * .000001;  
}

int main()
{
    ImageLoader* imageLoader = new ImageLoader();
    imageLoader->loadImage();

    vector<Mat> imgBuffer = imageLoader->getImgBuffer();


    double s = getCurrentTime();
    EGGraph egGraph;
    egGraph.buildEGGraph(imgBuffer);

    // double e = getCurrentTime();
    // cout << "using " << e - s << " s." << endl;
    
    // EGEdge** edge = egGraph.getEdgeMap();
    // int num = egGraph.getNodeNum();
    // ofstream of;
    // for(int i = 0; i < num; i++)
    // {
    //     for(int j = i + 1; j < num; j++)
    //     {
    //         of.open("../result/features/feature" + to_string(i) + "-" + to_string(j) + ".txt", ios::out);
    //         for(int k = 0; k < edge[i][j].keypoints1.size(); k++)
    //         {
    //             of << "(" << edge[i][j].keypoints1[k].pt.x << ", " << edge[i][j].keypoints1[k].pt.y << ") "
    //                << "(" << edge[i][j].keypoints2[k].pt.x << ", " << edge[i][j].keypoints2[k].pt.y << ")" << endl;                
    //         }
    //         of.close();
    //     }
    // }


    TrackManager trackManager;
    trackManager.merge(egGraph);
    trackManager.pruneTracks();

    vector< vector<TrackNode*> > tracks = trackManager.getTracks();
    ofstream of;
    for(int i = 0; i < tracks.size(); i++)
    {   
        of.open("../result/tracks/track" + to_string(i) + ".txt", ios::out);
        vector<TrackNode*> track = tracks[i];
        for(int j = 0; j < tracks[i].size(); j++)
        {
            of << track[j]->idx << " (" << tracks[i][j]->point.x 
            << ", " << tracks[i][j]->point.y << ")" << endl;
        }
        of.close();
    }
    cout << "track size is: " << tracks.size() << endl;

    delete imageLoader;
   
    // TrackManager trackManager;
    // trackManager.testMerge();

    waitKey();
}