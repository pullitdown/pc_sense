
#include "pc_sense/pc_op.hpp"





namespace pc_op{
    // template <typename PointT>
    // void projectCloud(const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
    // typename pcl::PointCloud<PointT>::Ptr& projected_cloud,
    // const Eigen::Affine3f& pose, 
    // const Eigen::Vector4f& plane_coefficients) 
    // {
    //     if(pose.linear() != Eigen::Matrix3f::Identity() || pose.translation() != Eigen::Vector3f::Zero())
    //     pcl::transformPointCloud(*cloud, *projected_cloud, pose);
        
    //     for (auto& point : projected_cloud->points) {
    //         float distance = plane_coefficients[0] * point.x + 
    //                         plane_coefficients[1] * point.y + 
    //                         plane_coefficients[2] * point.z + 
    //                         plane_coefficients[3];
    //         point.x -= plane_coefficients[0] * distance;
    //         point.y -= plane_coefficients[1] * distance;
    //         point.z -= plane_coefficients[2] * distance;
    //     }
    // }

}



    