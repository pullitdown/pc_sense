#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <visualization_msgs/Marker.h>
#include "pc_sense/pc_op.hpp"

enum cloudtype
{
plane_only,
floor_idx
};

class CloudProjector
{
public:
  template <typename T>
  T param(const ros::NodeHandle &nh, const std::string &name, const T &defaultValue,
          const bool silent = false)
  {
    if (nh.hasParam(name))
    {
      T v;
      nh.param<T>(name, v, defaultValue);
      if (!silent)
      {
        ROS_INFO_STREAM("Found parameter: " << name << ", value: " << v);
      }
      return v;
    }
    if (!silent)
    {
      ROS_WARN_STREAM("Cannot find value for parameter: " << name << ", assigning default: " << defaultValue);
    }
    return defaultValue;
  }

  template <typename PointT>
  sensor_msgs::PointCloud2 pcl2ros(const typename pcl::PointCloud<PointT>::Ptr &cloud)
  {
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header.frame_id = cloud->header.frame_id;
    output_msg.header.stamp = ros::Time::now();
    return output_msg;
  }

  CloudProjector()
  {
    nh_ = ros::NodeHandle("~");
    pub_0 = nh_.advertise<sensor_msgs::PointCloud2>("project_pc_0", 1);
    pub_1 = nh_.advertise<sensor_msgs::PointCloud2>("project_pc_1", 1);
    pub_2 = nh_.advertise<sensor_msgs::PointCloud2>("project_pc_2", 1);
    pub_3 = nh_.advertise<sensor_msgs::PointCloud2>("project_pc_3", 1);
    pub_4 = nh_.advertise<sensor_msgs::PointCloud2>("project_pc_4", 1);
    min_inlier_num = param<int>(nh_, "min_inlier_num", 5);
    max_plane_solver_condition_number = param<double>(nh_, "max_plane_solver_condition_number", 200);
    top_k_plane = param<int>(nh_, "top_k_plane", 4);
    cloud_type=static_cast<cloudtype>(param<int>(nh_, "cloud_type", 0));
    floor_plane_abcd << 0, 0, 1, 0;
    sub_ = nh_.subscribe("/svo/backend_points", 1, &CloudProjector::cloudCallback<pcl::PointXYZI>, this);
    marker_pub = nh_.advertise<visualization_msgs::Marker>("visualization_plane", 10);
  }

  void publishPlane(ros::Publisher &pub, const Eigen::Vector4d &position, const Eigen::Quaterniond &orientation)
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    // 设置平面的位置和方向
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = -position[3];
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.pose.orientation.w = 1;

    // 设置平面的尺寸
    marker.scale.x = 5.0;
    marker.scale.y = 5.0;
    marker.scale.z = 0.01; // Make it a thin plane

    // 设置颜色（绿色）
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    // 发布Marker消息
    pub.publish(marker);
  }

private:
  template <typename PointT>
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    // using cloud_local=typename pcl::PointCloud<PointT> ;
    typename pcl::PointCloud<PointT>::Ptr cloud(new typename pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *cloud);

    typename pcl::PointCloud<PointT>::Ptr projected_cloud(new typename pcl::PointCloud<PointT>);

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    Eigen::Vector4d centroid(0, 0, 1, 0);

    std::vector<Eigen::Matrix<double, 4, 1>> plane_abcds;
    std::vector<typename pcl::PointCloud<PointT>::Ptr> plane_clouds;
    std::vector<std::vector<int>> index_lists;
    std::vector<Eigen::Vector4d> plane_coefs;
    
    switch (cloud_type)
    {
    case plane_only:{
      if (!pc_op::plane_fitting_pc<PointT>(cloud, top_k_plane, plane_abcds, plane_clouds, min_inlier_num, max_plane_solver_condition_number))
      {
        ROS_WARN_STREAM("Failed to find planes " << plane_clouds.size());
      }
      break;}
    case floor_idx:{
      if (pc_op::plane_fitting_idx<PointT>(cloud, centroid, top_k_plane, index_lists, plane_coefs, min_inlier_num, max_plane_solver_condition_number))
      {
        Eigen::Vector4d plane_best;
        plane_best = plane_coefs.back();
        if (plane_best(2) < 0)
          plane_best *= -1;
        double error_angle = pc_op::calculateAngle(plane_best, this->floor_plane_abcd);
        double wight = 0.4;
        if (error_angle < 3.1412926 / 40)
          wight = 0.6;
        if (plane_best[3] - this->floor_plane_abcd[3] > 0.2)
          wight = 0.1;

        if (plane_best[3] - this->floor_plane_abcd[3] < -0.2)
          wight = 0.9;

        this->floor_plane_abcd = this->floor_plane_abcd * (1 - wight) + plane_best * wight;

        ROS_WARN_STREAM("find planes idx " << index_lists.size());
      }
      pc_op::plane_sieve_coef<PointT>(cloud, this->floor_plane_abcd, 0.3);
      pc_op::projectCloud<PointT>(cloud, projected_cloud, transform, centroid);
      pub_0.publish(pcl2ros<PointT>(cloud));
      pub_1.publish(pcl2ros<PointT>(projected_cloud));
      Eigen::Quaterniond q1 = Eigen::Quaterniond::Identity();
      publishPlane(marker_pub, this->floor_plane_abcd, q1);
      break;}
    default:{
      ROS_WARN_STREAM("no reference point cloud type");
      break;}
    }

    if (plane_clouds.size() > 3)
      pub_4.publish(pcl2ros<PointT>(plane_clouds[3]));
    if (plane_clouds.size() > 2)
      pub_3.publish(pcl2ros<PointT>(plane_clouds[2]));
    if (plane_clouds.size() > 1)
      pub_2.publish(pcl2ros<PointT>(plane_clouds[1]));
    // if(plane_clouds.size()>0)
  }

  ros::NodeHandle nh_;
  ros::Publisher pub_0;
  ros::Publisher pub_1;
  ros::Publisher pub_2;
  ros::Publisher pub_3;
  ros::Publisher pub_4;
  ros::Subscriber sub_;
  ros::Publisher marker_pub;
  cloudtype cloud_type;
  Eigen::Vector4d floor_plane_abcd;
  int min_inlier_num;
  double max_plane_solver_condition_number;
  int top_k_plane;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cloud_projector");
  CloudProjector projector;
  ros::spin();
  return 0;
}