#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

#include <Eigen/Geometry>
#include <sophus/se3.h>

using namespace std;

string est_poses_file = "../estimated.txt";
string gt_poses_file = "../groundtruth.txt";

typedef vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3> > SE3;

void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> pose1, 
                    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> pose2);


void read_poses(SE3& poses, string filename)
{
    fstream in(filename);
    if(in.is_open())
    {
        double ti, tx, ty, tz, qx, qy, qz, qw;
        while(in >> ti >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
        {
            Eigen::Vector3d t(tx, ty, tz);
            Eigen::Quaterniond q(qw, qx, qy, qz);
            Sophus::SE3 pose(q, t);
            poses.push_back(pose);
        }
    }
}


void compute_rmse(const SE3 es_poses, const SE3 gt_poses)
{
    int len = es_poses.size();
    Eigen::Matrix<double, 4, 4> es_transmat;
    Eigen::Matrix<double, 4, 4> gt_transmat;

    double mse = 0.0;
    Eigen::Matrix<double, 6, 1> minus_se3;
    
    for(int i = 0; i < len; i++)
    {
        Eigen::Matrix<double, 6, 1> es_se3 = es_poses[i].log();
        Eigen::Matrix<double, 6, 1> gt_se3 = gt_poses[i].log();
        minus_se3 << es_se3[0] - gt_se3[0],
                     es_se3[1] - gt_se3[1],
                     es_se3[2] - gt_se3[2],
                     es_se3[3] - gt_se3[3],
                     es_se3[4] - gt_se3[4],
                     es_se3[5] - gt_se3[5];
        mse += minus_se3.squaredNorm();
    }
    cout << sqrt(mse / (double)len) << endl;
}


int main()
{
    SE3 estimated_poses, groundtruth_poses;
    read_poses(estimated_poses, est_poses_file);
    read_poses(groundtruth_poses, gt_poses_file); 

    compute_rmse(estimated_poses, groundtruth_poses);

    DrawTrajectory(estimated_poses, groundtruth_poses);
}