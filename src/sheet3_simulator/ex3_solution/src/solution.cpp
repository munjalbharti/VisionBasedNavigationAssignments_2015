// This source code is intended for use in the teaching course "Vision-Based Navigation" at Technical University Munich only.
// Copyright 2015 Vladyslav Usenko, Joerg Stueckler, Technical University Munich

#include <ros/ros.h>
#include <uav_controller.hpp>

int main(int argc, char** argv) {

	ros::init(argc, argv, "ex4_solution");
	ros::NodeHandle nh;

	UAVController<double> u(nh);

	Sophus::SE3d desired_pose;
	desired_pose.translation() << 0,0,1;
	desired_pose.setQuaternion(Eigen::Quaterniond::Identity());

	u.setDesiredPose(desired_pose);

	ros::spin();

	return 0;
}

