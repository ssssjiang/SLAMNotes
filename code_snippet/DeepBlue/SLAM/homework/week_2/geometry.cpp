#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

Vector4d toHomogenious(const Vector3d p)
{
    return Vector4d(p(0), p(1), p(2), 1);
}

Vector3d toNonHomogenious(const Vector4d p)
{
    return Vector3d(p(0) / p(3), p(1) / p(3), p(2) / p(3));
}

int main()
{
    Quaterniond q1(0.55, 0.3, 0.2, 0.2);
    q1 = q1.normalized();
    Vector3d t1(0.7, 1.1, 0.2);

    Quaterniond q2(-0.1, 0.3, -0.7, 0.2);
    q2 = q2.normalized();

    Vector3d t2(-0.1, 0.4, 0.8);

    Vector3d p1(0.5, -0.1, 0.2);

    Matrix3d R = q1.toRotationMatrix();
    cout << "determinant(R): " << R.determinant() << endl;
    
    Matrix4d T;
    T << R(0, 0), R(0, 1), R(0, 2), t1(0),
         R(1, 0), R(1, 1), R(1, 2), t1(1),
         R(2, 0), R(2, 1), R(2, 2), t1(2),
         0, 0, 0, 1;
         
    Vector4d pw = T.inverse() * toHomogenious(p1);
    Vector3d p2 = q2.toRotationMatrix() * toNonHomogenious(pw) + t2;

    cout << "p2: " << p2 << endl;
}