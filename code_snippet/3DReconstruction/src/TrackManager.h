#ifndef TRACK_MANAGER_H
#define TRACK_MANAGER_H

#include "Track.h"
#include "TrackManager.h"
#include "EGGraph.h"
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


// #define vector<TrackNode*> Track

class TrackManager
{
private:
    vector< vector<TrackNode*> > tracks;


public:

    void mergeTracks(EGGraph egGraph);
    void merge(EGGraph egGraph);

    vector< vector<TrackNode*> > getTracks() const;

    void unite(TrackNode*& node1, TrackNode*& node2);
    TrackNode* findSet(TrackNode* node);
    
    int findTrackNode(TrackNode* trackNode) const;

    void testMerge();

    void pruneTracks();

};

#endif