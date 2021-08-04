#include "TrackManager.h"
#include "Utility.h"
// #include "ImagePair.h"
#include <map>
#include <utility>
#include <vector>
#include <queue>
#include <fstream>
#include <cstring>

#define MAX_SIZE 2000

using namespace std;


void TrackManager::testMerge()
{
    vector<TrackNode*> trackNodes[3];
    vector<Point2f> pts[3];
    pts[0].push_back(Point2f(171, 401)); pts[0].push_back(Point2f(168, 401));
    pts[0].push_back(Point2f(160, 401)); pts[0].push_back(Point2f(146, 378));

    pts[1].push_back(Point2f(146, 391)); pts[1].push_back(Point2f(144, 390));
    pts[1].push_back(Point2f(136, 389)); pts[1].push_back(Point2f(123, 387));

    pts[2].push_back(Point2f(138, 385)); pts[2].push_back(Point2f(129, 382));
    pts[2].push_back(Point2f(124.8, 379.2)); pts[2].push_back(Point2f(114, 378));

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            TrackNode* trackNode = new TrackNode();
            trackNode->idx = i;
            trackNode->point.x = (pts[i])[j].x;
            trackNode->point.y = (pts[i])[j].y;
            trackNode->rank = 0;
            trackNode->parent = trackNode;
            trackNodes[i].push_back(trackNode);
        }
    }

    for(int i = 0; i < 1; i++)
    {
        for(int j = i + 1; j < 3; j++)
        {
            for(int k = 0; k < 4; k++)
            {
                this->unite((trackNodes[i])[k], (trackNodes[j])[k]);
            }
        }
    }


    for(int i = 0; i < 1; i++)
    {
        for(int j = i + 1; j < 3; j++)
        {
            cout << "(" << i << ", " << j << ") beigins...\n";
            for(int k = 0; k < 4; k++)
            {
                // trackNode不在track中
                if(this->findTrackNode(trackNodes[i][k]) == -1)
                {
                    TrackNode* parent = this->findSet(trackNodes[i][k]);
                    int idx = this->findTrackNode(parent);
                    if(idx == -1)
                    {   // trackNode和父亲均不在track中,将两者同时插入track中
                        vector<TrackNode*> track;
                        if(parent != trackNodes[i][k])
                        {
                            track.push_back(parent);
                            track.push_back(trackNodes[i][k]);
                        }
                        else track.push_back(parent);
                        this->tracks.push_back(track);
                    }
                    else
                    {   // trackNode不在track中但是父亲在，之将trackNode本身插入track中
                        this->tracks[idx].push_back(trackNodes[i][k]);
                    }
                }

                if(this->findTrackNode(trackNodes[j][k]) == -1)
                {
                    TrackNode* parent = this->findSet(trackNodes[j][k]);
                    int idx = this->findTrackNode(parent);
                    if(idx == -1)
                    {   // trackNode和父亲均不在track中,将两者同时插入track中
                        vector<TrackNode*> track;
                        if(parent != trackNodes[j][k])
                        {
                            track.push_back(parent);
                            track.push_back(trackNodes[j][k]);
                        }
                        else track.push_back(parent);
                        this->tracks.push_back(track);
                    }
                    else
                    {  // trackNode不在track中但是父亲在，之将trackNode本身插入track中
                        this->tracks[idx].push_back(trackNodes[j][k]);
                    }
                }
            }
            cout << "(" << i << ", " << j << ") ends...\n" << endl;            
        }
    }

    ofstream of;
    for(int i = 0; i < tracks.size(); i++)
    {
        of.open("../result/testTracks/track" + to_string(i) + ".txt", ios::out);
        vector<TrackNode*> track = tracks[i];
        for(int j = 0; j < tracks[i].size(); j++)
        {
            of << track[j]->idx << " (" << tracks[i][j]->point.x 
            << ", " << tracks[i][j]->point.y << ")" << endl;
        }
        of.close();
    }
}


void TrackManager::unite(TrackNode*& node1, TrackNode*& node2)
{
    // cout << "Unite begins...\n";
    TrackNode* n1 = this->findSet(node1);
    TrackNode* n2 = this->findSet(node2);

    // if(n1 == n2) return;
    if(n1->idx == n2->idx && n1->point.x == n2->point.x && n1->point.y == n2->point.y)
        return;
    else
    {
        if(n1->rank < n2->rank) n1->parent = n2;
        else
        {
            n2->parent = n1;
            if(n1->rank == n2->rank) n1->rank++;
        }
    }
    // cout << "unite ends...\n";
}


TrackNode* TrackManager::findSet(TrackNode* node)
{
    if(node->parent == node) return node;
    else
    {
        return node->parent = findSet(node->parent);
    }
}



vector< vector<TrackNode*> > TrackManager::getTracks() const
{
    return this->tracks;
}


int TrackManager::findTrackNode(TrackNode* trackNode) const
{
    for(int i = 0; i < this->tracks.size(); i++)
    {
        for(int j = 0; j < this->tracks[i].size(); j++)
        {
            // if(trackNode == this->tracks[i][j])
            //     return i;
            if(trackNode->idx == this->tracks[i][j]->idx &&
               trackNode->point.x == this->tracks[i][j]->point.x &&
               trackNode->point.y == this->tracks[i][j]->point.y) return i;
        }
    }
    return -1;  // not found
}

void TrackManager::mergeTracks(EGGraph egGraph)
{
    EGEdge** edge = egGraph.getEdgeMap();
    int nodeNum = egGraph.getNodeNum();

    cout << "Merge tracks begins...\n";

    for(int i = 0; i < nodeNum; i++)
    {
        for(int j = i + 1; j < nodeNum; j++)
        {
            if(edge[i][j].isMatch)
            {
                for(int k = 0; k < edge[i][j].trackNodes1.size(); k++)
                {
                    this->unite(edge[i][j].trackNodes1[k], edge[i][j].trackNodes2[k]);
                }
            }
        }
    }
    cout << "Merge tracks ends...\n";
}

void TrackManager::merge(EGGraph egGraph)
{
    this->mergeTracks(egGraph);

    int nodeNum = egGraph.getNodeNum();
    EGEdge** edge = egGraph.getEdgeMap();

    for(int i = 0; i < nodeNum; i++)
    {
        for(int j = i + 1; j < nodeNum; j++)
        {
            cout << "(" << i << ", " << j << ") beigins...\n";
            if(edge[i][j].isMatch)
            {
                for(int k = 0; k < edge[i][j].trackNodes1.size(); k++)
                {
                    // trackNode不在track中
                    if(this->findTrackNode(edge[i][j].trackNodes1[k]) == -1)
                    {
                        TrackNode* parent = this->findSet(edge[i][j].trackNodes1[k]);
                        int idx = this->findTrackNode(parent);
                        if(idx == -1)
                        {   // trackNode和父亲均不在track中,将两者同时插入track中
                            vector<TrackNode*> track;
                            if(parent != edge[i][j].trackNodes1[k])
                            {
                                track.push_back(parent);
                                track.push_back(edge[i][j].trackNodes1[k]);
                            }
                            else track.push_back(parent);
                            this->tracks.push_back(track);
                        }
                        else
                        {   // trackNode不在track中但是父亲在，之将trackNode本身插入track中
                            this->tracks[idx].push_back(edge[i][j].trackNodes1[k]);
                        }
                    }

                    if(this->findTrackNode(edge[i][j].trackNodes2[k]) == -1)
                    {
                        TrackNode* parent = this->findSet(edge[i][j].trackNodes2[k]);
                        int idx = this->findTrackNode(parent);
                        if(idx == -1)
                        {   // trackNode和父亲均不在track中,将两者同时插入track中
                            vector<TrackNode*> track;
                            if(parent != edge[i][j].trackNodes2[k])
                            {
                                track.push_back(parent);
                                track.push_back(edge[i][j].trackNodes2[k]);
                            }
                            else track.push_back(parent);
                            this->tracks.push_back(track);
                        }
                        else
                        {   // trackNode不在track中但是父亲在，之将trackNode本身插入track中
                            this->tracks[idx].push_back(edge[i][j].trackNodes2[k]);
                        }
                    }
                }
            }
            cout << "(" << i << ", " << j << ") ends...\n" << endl;            
        }
    }
}


void TrackManager::pruneTracks()
{
    int count[MAX_SIZE];
    bool isErased;
    vector< vector<TrackNode*> >::iterator ite;
    int i = 0;
    for(ite = this->tracks.begin(); ite != this->tracks.end();)
    {
        memset(count, 0, sizeof(count));
        isErased = false;
        for(int j = 0; j < this->tracks[i].size(); j++)
        {
            if((++count[this->tracks[i][j]->idx]) > 1 || this->tracks[i].size() < 2) 
            {
                ite = this->tracks.erase(ite);
                isErased = true;
                break;
            }
        }
        if(!isErased) 
        {
            i++;
            ite++;
        }
    }
}