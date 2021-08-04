#ifndef TRACK_H
#define TRACK_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct TrackNode
{
    int idx;            // image index
    Point2f point;      // keypoint in image[idx]
    TrackNode* parent;  // parent node
    int rank;           // rank of track node

    // bool operator==(const TrackNode* node) const
    // {
    //     if(this->idx == node->idx 
    //         && this->point.x == node->point.x 
    //         && this->point.y == node->point.y)
    //     {
    //         return true;
    //     }

    //     return false;
    // }


    // bool operator!=(const TrackNode* node) const
    // {
    //     if(this->idx == node->idx 
    //         && this->point.x == node->point.x 
    //         && this->point.y == node->point.y)
    //     {
    //         return false;
    //     }

    //     return true;
    // }
};

// struct Track
// {
//     vector<TrackNode*> track; 
// };


#endif