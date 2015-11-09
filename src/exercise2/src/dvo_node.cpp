// This source code is intended for use in the teaching course "Vision-Based Navigation" in summer term 2015 at TU Munich only. 
// Copyright 2015 Robert Maier, Joerg Stueckler, TUM

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "cv_bridge/cv_bridge.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "opencv2/opencv.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/transforms.h>

#include <tf/transform_listener.h>


#include <sheet3_dvo/dvo.h>

#include <fstream>


cv::Mat grayRef, depthRef;
ros::Publisher pub_pointcloud;
Eigen::Matrix4f integeratedTransform=Eigen::Matrix4f::Identity();

void imagesToPointCloud( const cv::Mat& img_rgb, const cv::Mat& img_depth, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling = 1 ) {

  cloud->is_dense = true;
  cloud->height = img_depth.rows / downsampling;
  cloud->width = img_depth.cols / downsampling;
  cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 1.f );
  cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
  cloud->points.resize( cloud->height*cloud->width );

  const float invfocalLength = 1.f / 525.f;
  const float centerX = 319.5f;
  const float centerY = 239.5f;
  const float depthscale = 1.f;

  const float* depthdata = reinterpret_cast<const float*>( &img_depth.data[0] );
  const unsigned char* colordata = &img_rgb.data[0];
  int idx = 0;
  for( unsigned int y = 0; y < img_depth.rows; y++ ) {
    for( unsigned int x = 0; x < img_depth.cols; x++ ) {

      if( x % downsampling != 0 || y % downsampling != 0 ) {
        colordata += 3;
        depthdata++;
        continue;
      }

      pcl::PointXYZRGB& p = cloud->points[idx];

      if( *depthdata == 0.f || isnan(*depthdata) ) { //|| factor * (float)(*depthdata) > 10.f ) {
        p.x = std::numeric_limits<float>::quiet_NaN();
        p.y = std::numeric_limits<float>::quiet_NaN();
        p.z = std::numeric_limits<float>::quiet_NaN();
      }
      else {
        float xf = x;
        float yf = y;
        float dist = depthscale * (float)(*depthdata);
        p.x = (xf-centerX) * dist * invfocalLength;
        p.y = (yf-centerY) * dist * invfocalLength;
        p.z = dist;
      }

      depthdata++;

      int b = (*colordata++);

      int g = (*colordata++);
      int r = (*colordata++);

      int rgb = ( r << 16 ) + ( g << 8 ) + b;
      p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

      idx++;


    }
  }

}



void callback(const sensor_msgs::ImageConstPtr& image_rgb, const sensor_msgs::ImageConstPtr& image_depth)
{
  
	//std::cout << "Callback called" << std::endl;
    Eigen::Matrix3f cameraMatrix;
    cameraMatrix <<    525.0, 0.0, 319.5,
                         0.0, 525.0, 239.5,
                         0.0, 0.0, 1.0;
    
    cv_bridge::CvImageConstPtr img_rgb_cv_ptr = cv_bridge::toCvShare( image_rgb, "bgr8" );
    cv_bridge::CvImageConstPtr img_depth_cv_ptr = cv_bridge::toCvShare( image_depth, "32FC1" );
    
//    cv::imshow("img_rgb", img_rgb_cv_ptr->image );
//    cv::imshow("img_depth", 0.2*img_depth_cv_ptr->image );
//    cv::waitKey(10);
    
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
    cv::Mat grayCurInt;
    cv::cvtColor( img_rgb_cv_ptr->image.clone(), grayCurInt, CV_BGR2GRAY);
    cv::Mat grayCur;
    grayCurInt.convertTo(grayCur, CV_32FC1, 1.f/255.f);
    
    cv::Mat depthCur = img_depth_cv_ptr->image.clone();
    
    
    if( !grayRef.empty() )
    {
        //std::cout << "Alligining images" << std::endl ;
    	alignImages( transform, grayRef, depthRef, grayCur, depthCur, cameraMatrix );
    }
    grayRef = grayCur.clone();
    depthRef = depthCur.clone();
    
    // ROS_ERROR_STREAM( "transform: " << transform << std::endl );
    

    //  dump trajectory for evaluation 
    
    integeratedTransform = transform * integeratedTransform; 
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB > );
    imagesToPointCloud( img_rgb_cv_ptr->image, img_depth_cv_ptr->image, cloud );
    
    cloud->header = pcl_conversions::toPCL( image_rgb->header );
    
    cloud->header.frame_id = "/world";
    pcl::transformPointCloud( *cloud, *cloud, integeratedTransform );
     ROS_ERROR_STREAM("IntegeratedTransform" << integeratedTransform << std::endl); 
    pub_pointcloud.publish( *cloud );
    
            
}

int main(int argc, char** argv)
{
 std::cout << "Started-- in main" << std::endl ;
 ros::init(argc, argv, "sheet2_dvo_node");

  ros::NodeHandle nh("~");
  message_filters::Subscriber<sensor_msgs::Image> image_rgb_sub(nh, "/camera/rgb/image_color", 1);
  message_filters::Subscriber<sensor_msgs::Image> image_depth_sub(nh, "/camera/depth/image", 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_rgb_sub, image_depth_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  
  pub_pointcloud = nh.advertise< pcl::PointCloud< pcl::PointXYZRGB > >( "pointcloud", 1 );
  

  ros::Rate loop_rate(100);

  while (ros::ok())
  {
    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}




