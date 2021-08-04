## 第２题　熟悉Linux

1.　一般来说会使用三种方法：
(1) 使用sudo apt-get install命令．例如sudo apt-get install slam

(2) 使用dpkg命令安装deb包．例如：sudo dpkg -i xxx.deb

(3) 使用make install命令通过源代码安装．

这些软件一般安装在/usr/lib/目录下；对于第三方软件，有的会放在/opt/目录下.

2.　linux的环境变量是什么：每个用户登录系统后，都会有一个专用的运行环境。通常每个用户默认的环境都是相同的，这个默认环境实际上就是一组环境变量的定义。环境变量是一个具有特定名字的对象，它包含了一个或者多个应用程序所将使用到的信息。

如何定义新的环境变量：

(1) 可以在终端中输入　$PATH="$PATH":/xxx_path

(2) 修改/etc/profile文件，打开/etc/profile文件，在最下面添加：export PATH="$PATH:/xx_path"

(3)修改/etc目录下的environment文件中的PATH变量．

3.　根目录下包含一下目录：/bin, /boot, /dev, /etc, /home, /lib, /mnt, /opt, /proc, /root, /sbin, /srv, /sys, /tmp, /usr, /var.
home目录下存储普通用户的个人文件，root目录下包含系统启动时的一些核心文件，包括操作系统内核等，bin目录下包含系统启动时需要的执行文件．

４．

chmod u+rx a.sh　或
chmod 555 a.sh　或
chmod +rx a.sh

5.　chown xiang:xiang a.sh


## 第３题　SLAM综述文献阅读

1.定位(localization)，地图构建(mapping), 路径规划(path planning), 自动驾驶，高风险或导航困难的营救任务，增强现实，药物，可视化监督系统等．

2.为了在环境中进行精确的定位，一个精确的地图是必要的；但是为了构建一个好的地图，当不同的元素加入到地图中的时候能够精确地定位是必需的．

为场景构建地图有两方面的原因：首先，其他任务需要地图作为支持，例如，地图可以提供路径规划或者向操作员提供直观的可视化结果．其次，地图可以约束机器人状态估计中的错误．一方面，在定位时，随着误差的累积，场景会逐渐漂移；另一方面，当给定一个地图时，机器人可以通过重新访问已知区域重置它的位置错误(也叫作loop closure)．

3.　SLAM的发展可以分为三个阶段：

(1) classical age(1986-2004). 这一阶段中，主要使用概率公式解决SLAM问题，比如基于扩展的卡尔曼滤波器，Rao-Blackwellised Particle Filters, 极大似然估计等．

(2) algorithmic-analysis age(2004-2015). 在这一阶段，研究的主要是SLAM的基本属性，包括observability, convergence, consistency. 这一阶段中，稀疏性在对于构建高效的SLAM解决方法扮演了重要的角色，并且主要的SLAM库被开发出来．

(3) robust-perception age(2015-now)．这一阶段的SLAM满足四个关键要求：robust performance, high level understanding, resource awareness, task-driven, task-drive inference.

4.　[1] Durrant-Whyte H, Bailey T. Simultaneous localization and mapping: Part I[J]. IEEE Robotics & Automation Magazine, 2006, 13(2): 99-110

[2] Bailey T, Durrant-Whyte H. Simultaneous localization and mapping(SLAM): Part II[J]. IEEE Robotics & Automation Magazine, 2006, 13(3): 108-117

[3] Davison A J, Reid I D, Molton N D, et al. MonoSLAM: real-time single camera SLAM[j]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2007, 29(6): 1052-1067

## 第４题　CMake练习

见hello文件夹

## 第５题
１．

![](https://github.com/AIBluefisher/DeepBlue_SLAM/blob/master/week-1/orb_slam_dl.png)

２.

(a) 将生成一个名为ORB_SLAM2的动态链接库文件，6个可执行文件．

(b) include文件夹下包含头文件，src文件夹下包含include文件夹下的头文件实现的Ｃ源代码，Examples文件夹下包含Monocular, RGB-D, ROS, Stereo四个文件夹．

(c) 链接到了OpenCV, eigen3, Pangolin, DBow2, g2o这些库文件．

## 第６题

1.
![](https://github.com/AIBluefisher/DeepBlue_SLAM/blob/master/week-1/orb_slam_compile.png)


2.

在源代码路径下的CMakeLists.txt文件的最后添加以下cmake命令:

```
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/MySlam)

add_executable(myslam Examples/MySlam/myslam.cpp)

target_link_libraries(myslam ${PROJECT_NAME})

add_executable(myvideo Examples/MySlam/myvideo.cpp)

target_link_libraries(myvideo ${PROJECT_NAME})
```

3.

截图如图所示：

![](https://github.com/AIBluefisher/DeepBlue_SLAM/blob/master/week-1/myslam.png)

**体会:** 从运行情况来看，ORB-SLAM2用于做重建的话，感觉点还是太稀疏了；如果精度能够达到要求的话，感觉上是可以用于定位等应用的．

