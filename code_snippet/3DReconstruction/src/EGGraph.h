#ifndef EG_GRAPH_H
#define EG_GRAPH_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "Track.h"

using namespace std;
using namespace cv;


struct EGNode
{
    /*
    * node of Epipolar Geometry Graph
    */
    unsigned int idx;           // image index
    vector<KeyPoint> keypoints; // keypoints detected in image[idx]
    Mat descriptor;             // descriptor computed in image[idx]
};

struct EGEdge
{
    /*
    * edge of Epipolar Geometry Graph
    */
    vector<DMatch> matches;     // matches between image[s] and image[e]
    bool isMatch;               // false if image[s] and image[e] don't match
    Mat fundamentalMat;         // fundamental matrix
    Mat essentialMat;           // essentail matrix
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptor1;
    Mat descriptor2;
    
    vector<TrackNode*> trackNodes1;
    vector<TrackNode*> trackNodes2;


};

class EGGraph
{
    /*
    * Epipolar Geometry Graph
    */
private:
    vector<EGNode> nodeList;
    EGEdge** edgeMap;
    int nodeNum;

public:
    EGGraph();
    ~EGGraph();

    vector<EGNode> getEGNodeList() const;
    EGEdge** getEdgeMap() const;
    int getNodeNum() const;

    void buildEGGraph(vector<Mat> imgBuffer);    

    void estimateFundamentalMat(vector<Mat> imgBuffer, int i, int j);
    void estimateEssentialMat();

    void keypoints2TrackNodes(vector<KeyPoint> keypoints, vector<TrackNode*>& trackNodes, int idx);    
    void keypoints2TrackNodes(vector<KeyPoint> keypoints, vector<TrackNode*>& trackNodes, int idx, int x, int y);
    TrackNode* findTrackNode(int x, int y, TrackNode* trackNode) const;

private:
    void startMatch(vector<Mat> imgBuffer, int i, int j);
    void buildEGNode(vector<Mat> imgBuffer);
    void buildEGEdge(vector<Mat> imgBuffer);
    
};

#endif