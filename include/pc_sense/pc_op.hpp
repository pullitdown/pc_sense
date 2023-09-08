#pragma once
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree.h>
#include <Eigen/Dense>
#include <random>
#include <queue>
#include <unordered_map>
namespace pc_op
{

    /// @brief for store the plane information
    /// @tparam PointT
    template <typename PointT>
    struct plane_pc
    {

        typename pcl::PointCloud<PointT>::Ptr pc;
        Eigen::Vector4d plane_coef;
        int pc_size;
        double error;
        std::vector<int> idx_list; // Corresponding the fillter plane cloud idx in source point cloud and would not external call
        plane_pc(const typename pcl::PointCloud<PointT>::Ptr &ptr_,
                 const Eigen::Vector4d &plane_coef_,
                 const int pc_size_,
                 const double error_,
                 std::vector<int> &idx_list_) : pc(ptr_), plane_coef(plane_coef_), pc_size(pc_size_), error(error_)
        {
            idx_list = std::move(idx_list_);
        };
        plane_pc(const typename pcl::PointCloud<PointT>::Ptr &ptr_,
                 const Eigen::Vector4d &plane_coef_,
                 const int pc_size_,
                 const double error_) : pc(ptr_), plane_coef(plane_coef_), pc_size(pc_size_), error(error_){};
    };

    template <typename PointT>
    double point_plane_distance(const PointT &pt, const Eigen::Matrix<double, 4, 1> &plane)
    {
        return fabs(plane(0) * pt.x + plane(1) * pt.y + plane(2) * pt.z + plane(3));
    }

    template <typename PointT>
    void down_sampling_pc(typename pcl::PointCloud<PointT>::Ptr &cloud, const double leaf_size)
    {
        typename pcl::VoxelGrid<PointT> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel_filter.filter(*cloud);
    }

    /// @brief too slow ,cant run in realtime 
    /// @tparam PointT 
    /// @param cloud 
    /// @param leaf_size 
    /// @param new_points 
    /// @param Terminating_plane 
    template <typename PointT>
    void generate_points_in_empty_voxels(typename pcl::PointCloud<PointT>::Ptr &cloud, double leaf_size, typename pcl::PointCloud<PointT>::Ptr &new_points,
                                         Eigen::Vector4d Terminating_plane)
    {
        down_sampling_pc<PointT>(cloud, leaf_size * 2);
        Eigen::Vector3d direction = Terminating_plane.head<3>().normalized();
        double step = leaf_size;
        PointT new_point;
        for (size_t i = 0; i < cloud->points.size(); ++i)
        {

            new_point.x = cloud->points[i].x + step * direction[0];
            new_point.y = cloud->points[i].y + step * direction[1];
            new_point.z = cloud->points[i].z + step * direction[2];
            if (point_plane_distance<PointT>(new_point, Terminating_plane) > point_plane_distance<PointT>(cloud->points[i], Terminating_plane)) // judge the step direction
            {
                step = -step;
                new_point.x = cloud->points[i].x + step * direction[0];
                new_point.y = cloud->points[i].y + step * direction[1];
                new_point.z = cloud->points[i].z + step * direction[2];
            }
            while (point_plane_distance<PointT>(new_point, Terminating_plane) > 0.1)
            {
                // 在体素中心沿给定的方向生成一个新的点
                // 这里我们假设新的点就是体素中心点的一个小偏移
                new_point.x += step * direction[0];
                new_point.y += step * direction[1];
                new_point.z += step * direction[2];
                new_points->points.push_back(new_point);
            }
        }
        down_sampling_pc<PointT>(new_points, leaf_size);

        new_points->header = cloud->header;
        new_points->is_dense = false;
        new_points->width = new_points->points.size();
        new_points->height = 1;
    };

    template <typename PointT>
    void projectCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                      typename pcl::PointCloud<PointT>::Ptr &projected_cloud,
                      const Eigen::Affine3d &pose,
                      const Eigen::Vector4d &plane_coefficients)
    {
        // if(pose.linear() != Eigen::Matrix3f::Identity() || pose.translation() != Eigen::Vector3f::Zero())
        pcl::transformPointCloud(*cloud, *projected_cloud, pose);
        // std::cout<<"Projected cloud size: " << projected_cloud->size()<<std::endl;
        // std::cout<<" cloud size: " << cloud->size()<<std::endl;
        projected_cloud->resize(cloud->size());
        for (auto &point : projected_cloud->points)
        {
            float distance = plane_coefficients[0] * point.x +
                             plane_coefficients[1] * point.y +
                             plane_coefficients[2] * point.z +
                             plane_coefficients[3];
            point.x -= plane_coefficients[0] * distance;
            point.y -= plane_coefficients[1] * distance;
            point.z -= plane_coefficients[2] * distance;
        }
    };

    template <typename PointT>
    void getGridIdxDistance(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                            std::vector<double> &projected_distances,
                            std::vector<int> &grid_idx,
                            Eigen::Vector4d &plane_coefficients,
                            const Eigen::Vector2d &plane_size,
                            int invert_step = 10)
    {
        assert(plane_size[0] * plane_size[1] * invert_step * invert_step < INT_MAX);
        // assert(plane_coefficients.head<3>().norm() < 1.000001 && plane_coefficients.head<3>().norm() > 0.999999);
        
        projected_distances.reserve(cloud->size());
        grid_idx.reserve(cloud->size());
        double distance, x_plane, y_plane;
        double plane_dist_2_origin;
        int x_len = plane_size[0] * invert_step;
        int y_len = plane_size[1] * invert_step;
        int hx_len = x_len / 2;
        int hy_len = y_len / 2;
        Eigen::Vector3d origin_in_plane(0, 0, -plane_coefficients[3]);
        Eigen::Vector3d point_in_plane;
        Eigen::Vector3d new_x;
        Eigen::Vector3d new_y;
        PointT point_f=cloud->points[0];

        distance = plane_coefficients[0] * point_f.x+
                   plane_coefficients[1] * point_f.y+
                   plane_coefficients[2] * point_f.z+
                   plane_coefficients[3];

        point_in_plane[0] = (point_f.x - plane_coefficients[0] * distance);
        point_in_plane[1] = (point_f.y - plane_coefficients[1] * distance);
        point_in_plane[2] = (point_f.z - plane_coefficients[2] * distance);
        new_x = (point_in_plane - origin_in_plane).normalized();
        new_y = (plane_coefficients.head<3>().cross(new_x)).normalized();
        for (auto &point : cloud->points)
        {
            distance = plane_coefficients[0] * point.x +
                       plane_coefficients[1] * point.y +
                       plane_coefficients[2] * point.z +
                       plane_coefficients[3];
            projected_distances.emplace_back(distance);
            point_in_plane[0] = (point.x - plane_coefficients[0] * distance);
            point_in_plane[1] = (point.y - plane_coefficients[1] * distance);
            point_in_plane[2] = (point.z - plane_coefficients[2] * distance);
            x_plane = (point_in_plane - origin_in_plane).dot(new_x);
            y_plane = (point_in_plane - origin_in_plane).dot(new_y);
            grid_idx.emplace_back(static_cast<int>(x_plane * invert_step) + hx_len + (static_cast<int>(y_plane * invert_step) + hy_len) * x_len);
        }
    };

    ///@brief input the idxlist and pointcloud ,ouput all the pointcloud[idx] in idxlist
    template <typename PointT>
    void getPointCloudFromIdx(const typename pcl::PointCloud<PointT>::Ptr &incloud,
                              const std::vector<int> &idxlist, typename pcl::PointCloud<PointT>::Ptr &outcloud)
    {
        outcloud->points.resize(idxlist.size());
        for (int i = 0; i < idxlist.size(); i++)
        {
            outcloud->points[i] = incloud->points[idxlist[i]];
        }
    }

    /// @brief get plane_idx and distance ,return the same plane_idx max distance pointcloud idx
    template <typename PointT>
    void getTopPoint(
        typename pcl::PointCloud<PointT>::Ptr &cloud,
        typename pcl::PointCloud<PointT>::Ptr &outcloud,
        Eigen::Vector4d &plane_coefficients,
        const Eigen::Vector2d &plane_size,
        int invert_step = 10)
    {

        plane_coefficients/= plane_coefficients.head(3).norm();
        std::vector<double> projected_distances;
        std::vector<int> grid_idx;
        std::vector<int> top_points;
        getGridIdxDistance<PointT>(cloud,
                                   projected_distances,
                                   grid_idx,
                                   plane_coefficients,
                                   plane_size,
                                   invert_step);
        std::unordered_map<int, std::pair<int, double>> my_map;
        double value;
        int key;
        for (int idx = 0; idx < cloud->points.size(); idx++)
        {

            value = projected_distances[idx];
            auto it = my_map.find(grid_idx[idx]);
            if (it != my_map.end())
                it->second.second = std::max(it->second.second, value);
            else
                my_map.insert({grid_idx[idx], {idx, value}});
        }

        for (auto it = my_map.begin(); it != my_map.end(); it++)
        {
            top_points.push_back(it->second.first);
        }

        getPointCloudFromIdx<PointT>(cloud, top_points, outcloud);
        outcloud->header = cloud->header;
        outcloud->is_dense = false;
        outcloud->width = outcloud->points.size();
        outcloud->height = 1;
    }

    /// @brief calculate the angle between two planes
    /// @param n1_ first plane's coefficients
    /// @param n2_
    /// @return the angle liks 3.1415926=pi
    double calculateAngle(const Eigen::Vector4d &n1_, const Eigen::Vector4d &n2_)
    {

        double cosTheta = n1_.head<3>().dot(n2_.head<3>()) / (n1_.head<3>().norm() * n2_.head<3>().norm());
        double angle = std::acos(cosTheta);
        return angle;
    }

    static inline double point_to_plane_distance(const Eigen::Vector3d &point, const Eigen::Vector4d &abcd)
    {
        return point.dot(abcd.head(3)) + abcd(3);
    };

    bool fit_plane(const std::vector<Eigen::Vector3d> &feats, Eigen::Vector4d &abcd, double cond_thresh,
                   bool cond_check = true)
    {

        // Check whether we have enough constraints
        if (feats.size() < 3)
        {
            // PRINT_DEBUG("[PLANE-FIT]: Not having enough constraint for plane fitting! (%d )\n", feats.size());
            return false;
        }

        // Linear system
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero((int)feats.size(), 3);
        Eigen::VectorXd b = -Eigen::VectorXd::Ones((int)feats.size(), 1);
        for (size_t i = 0; i < feats.size(); i++)
        {
            A.row(i) = feats.at(i);
        }

        // Check condition number to avoid singularity
        if (cond_check)
        {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
            double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
            if (cond > cond_thresh)
            {
                // PRINT_DEBUG("[PLANE-FIT]: The condition number is too big!! (%.2f > %.2f)\n", cond, cond_thresh);
                return false;
            }
        }

        // Solve plane a b c d by QR decomposition
        // ax + by + cz + d = 0
        abcd.head(3) = A.colPivHouseholderQr().solve(b);
        // d = 1.0
        abcd(3) = 1.0;
        // Divide the whole vector by the norm of the normal direction of the plane
        abcd /= abcd.head(3).norm();

        // Check if the plane is invalid (depth of plane near zero)
        double dist_thresh = 0.02;
        Eigen::Vector3d cp = -abcd.head(3) * abcd(3);
        return (cp.norm() > dist_thresh);
    };

    template <typename PointT>
    bool fit_plane_pc(const typename pcl::PointCloud<PointT>::Ptr &pc, Eigen::Vector4d &abcd, double cond_thresh,
                      bool cond_check = true)
    {
        std::vector<Eigen::Vector3d> feats;
        feats.reserve(pc->size());
        for (size_t i = 0; i < pc->size(); i++)
        {
            feats.emplace_back(pc->points[i].x, pc->points[i].y, pc->points[i].z);
        }
        return fit_plane(feats, abcd, cond_thresh, cond_check);
    };

    template <typename PointT, typename PointT_>
    double norm_(const PointT &a, const PointT_ &b)
    {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    }

    // fitting ktop plane from point cloud
    template <typename PointT>
    bool plane_fitting_pc(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                          const int top_k_plane,
                          std::vector<Eigen::Matrix<double, 4, 1>> &plane_abcds,
                          std::vector<typename pcl::PointCloud<PointT>::Ptr> &plane_clouds,
                          int min_inlier_num,
                          double max_plane_solver_condition_number)
    {
        // condition plane data
        auto cmp = [&](const plane_pc<PointT> &a, const plane_pc<PointT> &b) -> bool
        { return (a.pc_size > b.pc_size) || (a.pc_size == b.pc_size && a.error < b.error); };
        std::priority_queue<plane_pc<PointT>, std::vector<plane_pc<PointT>>, decltype(cmp)> pq(cmp); // min top heap ,the min item will be pop

        const int ransac_solver_feat_num = 5;
        const int max_iter_num = 200;
        const double min_inlier_ratio = 0.10;
        const double max_error_threshold = 0.05;
        const double min_distance_between_points = 0.05;
        const size_t min_feat_on_plane_num_threshold = std::max(min_inlier_num, (int)((double)cloud->points.size() * min_inlier_ratio));
        std::mt19937 rand_gen(8888);

        // If the features are too few, we skip them
        if ((int)cloud->points.size() < min_inlier_num)
        {
            return false;
        }

        for (size_t n = 0; n < max_iter_num; n++)
        {

            // Create a random set of features
            typename pcl::PointCloud<PointT>::Ptr random_pc(new typename pcl::PointCloud<PointT>);
            typename pcl::PointCloud<PointT>::Ptr ransac_pc(new typename pcl::PointCloud<PointT>);
            pcl::copyPointCloud(*cloud, *random_pc);
            std::shuffle(random_pc->points.begin(), random_pc->points.end(), rand_gen);
            // Loop until we have enough points or when we run out of option
            auto it = random_pc->points.begin();
            while (ransac_pc->points.size() < ransac_solver_feat_num && it != random_pc->points.end())
            {

                // Push back directly for the first one
                if (ransac_pc->points.empty())
                {
                    ransac_pc->points.push_back(*it);
                }
                else
                {
                    // Check distance between this point and the current set
                    bool good_feat = true;

                    for (const auto &pf : ransac_pc->points)
                    {
                        if (norm_(pf, *it) < min_distance_between_points)
                        {
                            good_feat = false;
                            break;
                        }
                    }
                    // If pass distance check, add it to the ransac set
                    if (good_feat)
                    {
                        ransac_pc->points.push_back(*it);
                    }
                }
                it++;
            }

            // If not enough features the just return
            if (ransac_pc->points.size() != ransac_solver_feat_num)
            {

                return false;
            }

            // Try to solve the plane when we have enough features
            Eigen::Vector4d plane_abcd_;
            if (fit_plane_pc<PointT>(ransac_pc, plane_abcd_, max_plane_solver_condition_number))
            {

                // Check the other p_FinG and check the number of iniliers
                double inlier_avg_error = 0.0;
                std::vector<Eigen::Vector3d> inliers;
                for (auto &point : random_pc->points)
                {
                    double error = point_plane_distance(point, plane_abcd_);
                    if (std::abs(error) < max_error_threshold)
                    {
                        ransac_pc->points.push_back(point);
                        inlier_avg_error += std::abs(error);
                    }
                }
                inlier_avg_error /= (double)ransac_pc->points.size();

                // If pass inlier number threshold and error threshold then the plane is good to be accepted
                // The set is better if it has more inliers or if it has the same number and smaller error!
                bool valid_set = (ransac_pc->points.size() > min_feat_on_plane_num_threshold && inlier_avg_error < max_error_threshold);

                if (valid_set)
                {
                    pq.emplace(ransac_pc, plane_abcd_, ransac_pc->points.size(), inlier_avg_error);
                    if (pq.size() > top_k_plane)
                        pq.pop();
                }
            }
        } // end of for
        plane_abcds.reserve(pq.size());
        plane_clouds.reserve(pq.size());
        // Check that we have a good set of inliers
        bool ss = !pq.empty();
        int i = 0;
        while (!pq.empty())
        {

            auto item = pq.top();
            if (fit_plane_pc<PointT>(item.pc, item.plane_coef, max_plane_solver_condition_number, false))
            {

                // Calculate measurement sigma
                Eigen::ArrayXd errors = Eigen::ArrayXd::Zero((int)item.pc->size(), 1);
                for (auto &point : item.pc->points)
                {
                    errors(i) = point_plane_distance(point, item.plane_coef);
                }
                item.pc->header = cloud->header;
                item.pc->is_dense = false;
                item.pc->width = item.pc->points.size();
                item.pc->height = 1;
                double inlier_std = std::sqrt((errors - errors.mean()).square().sum() / (double)(errors.size() - 1));
                double inlier_err = errors.abs().mean();
                item.error = inlier_err;
                // Debug print
                plane_abcds.emplace_back(item.plane_coef);
                plane_clouds.emplace_back(item.pc);
                ++i;
            }

            pq.pop();
            // Further optimize the initial value using the inlier set
        }
        return ss;
    };

    bool plane_fitting(std::vector<Eigen::Vector3d> &feats, Eigen::Vector4d &plane_abcd, int min_inlier_num,
                       double max_plane_solver_condition_number)
    {
        // RANSAC params for plane fitting
        // TODO: read these from parameter file....
        const int ransac_solver_feat_num = 5;
        const int max_iter_num = 200;
        const double min_inlier_ratio = 0.80;
        const double max_error_threshold = 0.05;
        const double min_distance_between_points = 0.05;
        const size_t min_feat_on_plane_num_threshold = std::max(min_inlier_num, (int)((double)feats.size() * min_inlier_ratio));
        std::mt19937 rand_gen(8888);

        // If the features are too few, we skip them
        if ((int)feats.size() < min_inlier_num)
        {

            return false;
        }

        // Solve by ransace if we have enough features
        double best_error = -1;
        std::vector<Eigen::Vector3d> best_inliers;
        for (size_t n = 0; n < max_iter_num; n++)
        {

            // Create a random set of features
            std::vector<Eigen::Vector3d> feat_vec_copy = feats;
            std::vector<Eigen::Vector3d> ransac_set;
            std::shuffle(feat_vec_copy.begin(), feat_vec_copy.end(), rand_gen);

            // Loop until we have enough points or when we run out of option
            auto it = feat_vec_copy.begin();
            while (ransac_set.size() < ransac_solver_feat_num && it != feat_vec_copy.end())
            {

                // Push back directly for the first one
                if (ransac_set.empty())
                {
                    ransac_set.push_back(*it);
                }
                else
                {
                    // Check distance between this point and the current set
                    bool good_feat = true;
                    Eigen::Vector3d p_FinG = (*it);
                    for (const auto &pf : ransac_set)
                    {
                        if ((pf - p_FinG).norm() < min_distance_between_points)
                        {
                            good_feat = false;
                            break;
                        }
                    }
                    // If pass distance check, add it to the ransac set
                    if (good_feat)
                    {
                        ransac_set.push_back(*it);
                    }
                }
                it++;
            }

            // If not enough features the just return
            if (ransac_set.size() != ransac_solver_feat_num)
            {

                return false;
            }

            // Try to solve the plane when we have enough features
            if (fit_plane(ransac_set, plane_abcd, max_plane_solver_condition_number))
            {

                // Check the other p_FinG and check the number of iniliers
                double inlier_avg_error = 0.0;
                std::vector<Eigen::Vector3d> inliers;
                for (auto &feat : feats)
                {
                    double error = point_to_plane_distance(feat, plane_abcd);
                    if (std::abs(error) < max_error_threshold)
                    {
                        inliers.push_back(feat);
                        inlier_avg_error += std::abs(error);
                    }
                }
                inlier_avg_error /= (double)inliers.size();

                // If pass inlier number threshold and error threshold then the plane is good to be accepted
                // The set is better if it has more inliers or if it has the same number and smaller error!
                bool valid_set = (inliers.size() > min_feat_on_plane_num_threshold && inlier_avg_error < max_error_threshold);
                bool better_set =
                    ((best_inliers.size() < inliers.size()) || (best_inliers.size() == inliers.size() && inlier_avg_error < best_error));
                if (valid_set && better_set)
                {
                    best_inliers = inliers;
                    best_error = inlier_avg_error;
                }
            }
        } // end of for

        // Check that we have a good set of inliers
        if (!best_inliers.empty())
        {

            // Further optimize the initial value using the inlier set
            if (fit_plane(best_inliers, plane_abcd, max_plane_solver_condition_number, false))
            {

                // Calculate measurement sigma
                Eigen::ArrayXd errors = Eigen::ArrayXd::Zero((int)best_inliers.size(), 1);
                for (size_t i = 0; i < best_inliers.size(); i++)
                {
                    errors(i) = point_to_plane_distance(best_inliers.at(i), plane_abcd);
                }
                double inlier_std = std::sqrt((errors - errors.mean()).square().sum() / (double)(errors.size() - 1));
                double inlier_err = errors.abs().mean();

                // Debug print
                feats = best_inliers;
                return true;
            }
        }
        return false;
    };

    template <typename PointT>
    void ransacDetectPlane(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                           std::vector<Eigen::Vector4d> &planes_coef);

    template <typename PointT>
    void plane_sieve_coef(typename pcl::PointCloud<PointT>::Ptr &cloud,
                          const Eigen::Vector4d &plane_coef,
                          const double error_threshold = 0.05)
    {

        cloud->erase(std::remove_if(cloud->begin(), cloud->end(),
                                    [&plane_coef, &error_threshold](PointT &a)
                                    { return point_plane_distance<PointT>(a, plane_coef) < error_threshold; }),
                     cloud->end());
    }

    template <typename PointT>
    void plane_sieve_idx(typename pcl::PointCloud<PointT>::Ptr &cloud,
                         const std::vector<int> &index_list)
    {
        std::vector<bool>
            cloud_sieve(cloud->size(), true);
        for (size_t i = 0; i < index_list.size(); i++)
        {
            cloud_sieve[index_list[i]] = false;
        }
        cloud->erase(std::remove_if(cloud->begin(), cloud->end(),
                                    [&cloud_sieve, n = 0](PointT &a) mutable
                                    { return cloud_sieve[n++]; }),
                     cloud->end());
    }

    /// @brief filter the plane point idx in cloud
    /// @param[in] cloud source cloud
    /// @param[in] plane_coef which plane point would be filter
    /// @param[out] indexlists the output idxlists
    template <typename PointT>
    bool plane_fitting_idx(const typename pcl::PointCloud<PointT>::ConstPtr &src_cloud,
                           const Eigen::Vector4d &planes_coef,
                           int top_k,
                           std::vector<std::vector<int>> &index_lists,
                           std::vector<Eigen::Vector4d> &plane_coefs,
                           int min_inlier_num,
                           double max_plane_solver_condition_number)
    {
        auto cmp = [&](const plane_pc<PointT> &a, const plane_pc<PointT> &b) -> bool
        { return (a.pc_size > b.pc_size) || (a.pc_size == b.pc_size && a.error < b.error); };
        std::priority_queue<plane_pc<PointT>, std::vector<plane_pc<PointT>>, decltype(cmp)> pq(cmp); // min top heap ,the min item will be pop

        const int ransac_solver_feat_num = 5;
        const int max_iter_num = 200;
        const double min_inlier_ratio = 0.10;
        const double max_error_threshold = 0.05;
        const double min_distance_between_points = 0.05;
        const size_t min_feat_on_plane_num_threshold = std::max(min_inlier_num, (int)((double)src_cloud->points.size() * min_inlier_ratio));
        std::mt19937 rand_gen(8888);

        // If the features are too few, we skip them
        if ((int)src_cloud->points.size() < min_inlier_num)
        {
            return false;
        }

        for (size_t n = 0; n < max_iter_num; n++)
        {
            typename pcl::PointCloud<PointT>::Ptr ransac_pc(new typename pcl::PointCloud<PointT>);
            std::vector<int> v(src_cloud->points.size());
            std::iota(v.begin(), v.end(), 0);
            std::shuffle(v.begin(), v.end(), rand_gen);
            // Loop until we have enough points or when we run out of option
            auto it = v.begin();
            while (ransac_pc->points.size() < ransac_solver_feat_num && it != v.end())
            {

                // Push back directly for the first one
                if (ransac_pc->points.empty())
                {
                    ransac_pc->points.push_back(src_cloud->points[*it]);
                }
                else
                {
                    // Check distance between this point and the current set
                    bool good_feat = true;

                    for (const auto &pf : ransac_pc->points)
                    {
                        if (norm_(pf, src_cloud->points[*it]) < min_distance_between_points)
                        {
                            good_feat = false;
                            break;
                        }
                    }
                    // If pass distance check, add it to the ransac set
                    if (good_feat)
                    {
                        ransac_pc->points.push_back(src_cloud->points[*it]);
                    }
                }
                it++;
            }

            // If not enough features the just return
            if (ransac_pc->points.size() != ransac_solver_feat_num)
            {

                return false;
            }

            // Try to solve the plane when we have enough features
            Eigen::Vector4d plane_abcd_;
            std::vector<int> ransac_idx;
            if (fit_plane_pc<PointT>(ransac_pc, plane_abcd_, max_plane_solver_condition_number))
            {
                if (std::abs(calculateAngle(plane_abcd_, planes_coef)) < 3.1415926 / 10)
                {
                    double inlier_avg_error = 0.0;
                    for (auto &idx : v)
                    {
                        double error = point_plane_distance(src_cloud->points[idx], plane_abcd_);
                        if (std::abs(error) < max_error_threshold)
                        {
                            ransac_pc->points.push_back(src_cloud->points[idx]);
                            ransac_idx.push_back(idx);
                            inlier_avg_error += std::abs(error);
                        }
                    }
                    inlier_avg_error /= (double)ransac_pc->points.size();

                    bool valid_set = (ransac_pc->points.size() > min_feat_on_plane_num_threshold && inlier_avg_error < max_error_threshold);

                    if (valid_set)
                    {
                        pq.emplace(ransac_pc, plane_abcd_, ransac_pc->points.size(), inlier_avg_error, ransac_idx);
                        if (pq.size() > top_k)
                            pq.pop();
                    }
                }
            }
        } // end of for

        // Check that we have a good set of inliers

        if (!pq.empty())
        {
            auto item = pq.top();

            if (fit_plane_pc<PointT>(item.pc, item.plane_coef, max_plane_solver_condition_number, false))
            {
                if (std::abs(calculateAngle(item.plane_coef, planes_coef)) < 3.1415926 / 25)
                {

                    index_lists.push_back(item.idx_list);
                    plane_coefs.push_back(item.plane_coef);
                }
            }
            pq.pop();
            // Further optimize the initial value using the inlier set
        }

        return index_lists.size() > 0;
    }

}