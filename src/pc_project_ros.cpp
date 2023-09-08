#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/common/transforms.h>
#include <visualization_msgs/Marker.h>
#include "pc_sense/pc_op.hpp"
#include <Eigen/Dense>

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
    marker_pub = nh_.advertise<visualization_msgs::Marker>("visualization_plane", 10);
    marker_array_pub = nh_.advertise<visualization_msgs::MarkerArray>("visualization_lines", 1);
    min_inlier_num = param<int>(nh_, "min_inlier_num", 5);
    max_plane_solver_condition_number = param<double>(nh_, "max_plane_solver_condition_number", 200);
    top_k_plane = param<int>(nh_, "top_k_plane", 4);
    cloud_type = static_cast<cloudtype>(param<int>(nh_, "cloud_type", 0));
    leaf_size = param<double>(nh_, "leaf_size", 0.05);
    invert_step = param<int>(nh_, "invert_step", 10);
    plane_size[0] = param<int>(nh_, "plane_size_x", 10);
    plane_size[1] = param<int>(nh_, "plane_size_y", 10);
    frame_id = param<std::string>(nh_, "frame_id", "world");

    floor_plane_abcd << 0, 0, 1, 0;
    floor_fixed = 0;

    sub_ = nh_.subscribe("/svo/backend_points", 1, &CloudProjector::cloudCallback<pcl::PointXYZI>, this);
  }

  void publishPlane(ros::Publisher &pub, const Eigen::Vector4d &position)
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = this->frame_id;
    marker.header.stamp = ros::Time::now();
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    // 设置平面的位置和方向
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = -position[3];

    Eigen::Vector3d z = position.head<3>().normalized();
    Eigen::Vector3d p(1, 0, 0);
    Eigen::Vector3d p_alternative(0, 1, 0);
    if (std::fabs(z.dot(p)) > std::fabs(z.dot(p_alternative)))
      p = p_alternative;
    Eigen::Vector3d y = z.cross(p); // make sure gravity is not in x direction
    y.normalize();
    const Eigen::Vector3d x = y.cross(z);
    Eigen::Matrix3d C_imu_world; // world unit vectors in imu coordinates
    C_imu_world.col(0) = x;
    C_imu_world.col(1) = y;
    C_imu_world.col(2) = z;
    Eigen::Quaternion<double> eigen_quaternion(C_imu_world);
    
    marker.pose.orientation.x = eigen_quaternion.x();
    marker.pose.orientation.y = eigen_quaternion.y();
    marker.pose.orientation.z = eigen_quaternion.z();
    marker.pose.orientation.w = eigen_quaternion.w();
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

  template <typename PointT>
  void publishMarkersArray(const typename pcl::PointCloud<PointT>::Ptr &cloud, const Eigen::Vector4d &plane)
  {

    visualization_msgs::MarkerArray marker_array;

    // 遍历点云
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
      PointT &point = cloud->points[i];

      // 计算点到平面的距离
      double distance = plane[0] * point.x + plane[1] * point.y + plane[2] * point.z + plane[3];

      // 计算投影点
      PointT proj_point;
      proj_point.x = point.x - distance * plane[0];
      proj_point.y = point.y - distance * plane[1];
      proj_point.z = point.z - distance * plane[2];

      // 创建一个Marker
      visualization_msgs::Marker marker;
      marker.header.stamp = ros::Time::now(); // 假设你的点云是在base_link坐标系下的
      marker.header.frame_id = cloud->header.frame_id;
      marker.id = i;
      // 使用当前时间作为时间戳
      marker.type = visualization_msgs::Marker::LINE_LIST;
      marker.scale.x = 0.08; // 设置线宽
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;

      // 设置Marker的两个点
      geometry_msgs::Point p1, p2;
      p1.x = point.x;
      p1.y = point.y;
      p1.z = point.z;
      p2.x = proj_point.x;
      p2.y = proj_point.y;
      p2.z = proj_point.z;

      marker.points.push_back(p1);
      marker.points.push_back(p2);

      // 将Marker添加到MarkerArray中
      marker_array.markers.push_back(marker);
    }
    this->marker_array_pub.publish(marker_array);
  }

private:
  template <typename PointT>
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    // using cloud_local=typename pcl::PointCloud<PointT> ;
    typename pcl::PointCloud<PointT>::Ptr cloud(new typename pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *cloud);

    typename pcl::PointCloud<PointT>::Ptr projected_cloud(new typename pcl::PointCloud<PointT>);
    typename pcl::PointCloud<PointT>::Ptr top_cloud(new typename pcl::PointCloud<PointT>);
    // 计算点云的质心

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    Eigen::Vector4d centroid(0, 0, 1, 0);

    std::vector<Eigen::Matrix<double, 4, 1>> plane_abcds;
    std::vector<typename pcl::PointCloud<PointT>::Ptr> plane_clouds;
    std::vector<std::vector<int>> index_lists;
    std::vector<Eigen::Vector4d> plane_coefs;

    switch (cloud_type)
    {
    case plane_only:
    {
      if (!pc_op::plane_fitting_pc<PointT>(cloud, top_k_plane, plane_abcds, plane_clouds, min_inlier_num, max_plane_solver_condition_number))
      {
        ROS_WARN_STREAM("Failed to find planes " << plane_clouds.size());
      }
      break;
    }
    case floor_idx:
    {
      if (!floor_fixed && pc_op::plane_fitting_idx<PointT>(cloud, centroid, top_k_plane, index_lists, plane_coefs, min_inlier_num, max_plane_solver_condition_number))
      {

        Eigen::Vector4d plane_best;
        plane_best = plane_coefs.back();
        if (plane_best.head<3>().dot(this->floor_plane_abcd.head<3>()) < 0)
          plane_best *= -1;
        double error_angle = std::abs(pc_op::calculateAngle(plane_best, this->floor_plane_abcd));
        double wight = 0.4;
        if (error_angle < 3.1412926 / 40)
          wight = 0;
        else if (error_angle < 3.1412926 / 35)
          wight = 0.6;
        else if (plane_best[3] - this->floor_plane_abcd[3] > 0.2)
          wight = 0.1;
        else if (plane_best[3] - this->floor_plane_abcd[3] < -0.2)
          wight = 0.9;
        
        this->floor_plane_abcd.head<3>() = (this->floor_plane_abcd.head<3>() * (1 - wight) +plane_best.head<3>() * wight).normalized();
        this->floor_plane_abcd[3] = this->floor_plane_abcd[3] * (1 - wight) + plane_best[3] * wight;

        ROS_WARN_STREAM("find planes idx " << index_lists.size());
      }
      pc_op::plane_sieve_coef<PointT>(cloud, this->floor_plane_abcd, 0.3);

      Eigen::Affine3d pose = Eigen::Affine3d::Identity();
      pc_op::projectCloud<PointT>(cloud, projected_cloud, pose, this->floor_plane_abcd);

      pc_op::getTopPoint<PointT>(cloud, top_cloud, this->floor_plane_abcd, this->plane_size, this->invert_step);
      // ROS_WARN_STREAM("before downsample" << projected_cloud->points.size());
      // pc_op::down_sampling_pc<PointT>(projected_cloud, this->leaf_size);
      // ROS_WARN_STREAM("after downsample " << projected_cloud->points.size());
      // pc_op::generate_points_in_empty_voxels<PointT>(cloud, this->leaf_size, projected_cloud, this->floor_plane_abcd);

      pub_0.publish(pcl2ros<PointT>(cloud));
      pub_1.publish(pcl2ros<PointT>(projected_cloud));
      pub_2.publish(pcl2ros<PointT>(top_cloud));
      publishPlane(marker_pub, this->floor_plane_abcd);
      publishMarkersArray<PointT>(top_cloud, this->floor_plane_abcd);
      break;
    }
    default:
    {
      ROS_WARN_STREAM("no reference point cloud type");
      break;
    }
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
  ros::Publisher marker_array_pub;
  cloudtype cloud_type;
  Eigen::Vector4d floor_plane_abcd;
  int min_inlier_num;
  double max_plane_solver_condition_number;
  int top_k_plane;
  bool floor_fixed;
  double leaf_size;
  int invert_step;
  Eigen::Vector2d plane_size;
  std::string frame_id;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cloud_projector");
  CloudProjector projector;
  ros::spin();
  return 0;
}