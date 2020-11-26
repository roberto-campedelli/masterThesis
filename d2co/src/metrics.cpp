#include "metrics.h"

#include "cv_ext/cv_ext.h"
#include <ceres/rotation.h>
#include <opencv2/ml.hpp>

#include <vector>

MaxRtErrorsMetric::MaxRtErrorsMetric( double max_rot_err_deg, double max_t_err_cm ) :
    max_rot_err_deg_(max_rot_err_deg),
    max_t_err_cm_(max_t_err_cm)
{}

void MaxRtErrorsMetric::setGroundTruth(Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec)
{
  gt_r_quat_ = gt_r_quat;
  gt_t_vec_ = gt_t_vec;
}

bool MaxRtErrorsMetric::performTest(Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, bool symmetric_object )
{
  Eigen::Matrix3d rot_mat = r_quat.toRotationMatrix(), gt_rot_mat = gt_r_quat_.toRotationMatrix();

  double rot_err = 180.0*cv_ext::rotationDist(rot_mat, gt_rot_mat)/M_PI,
         t_err = 100.0*(t_vec - gt_t_vec_).norm();

  return ( rot_err < max_rot_err_deg_ && t_err < max_t_err_cm_ );
}

Projection2DMetric::Projection2DMetric( const RasterObjectModel3DPtr &obj_model_ptr, double max_proj_err_pix ) :
    obj_model_ptr_(obj_model_ptr),
    max_proj_err_pix_(max_proj_err_pix)
{}

void Projection2DMetric::setGroundTruth(Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec)
{
  obj_model_ptr_->setModelView( gt_r_quat, gt_t_vec );
  obj_model_ptr_->projectVertices(gt_proj_vtx_ );

  gt_proj_nd_vtx_.clear();
  gt_proj_nd_vtx_.reserve(gt_proj_vtx_.size());

  // Remove duplicates
  bool add_elem;
  for( const auto &new_elem : gt_proj_vtx_ )
  {
    add_elem = true;
    for( auto &cur_elem : gt_proj_nd_vtx_ )
    {
      if( new_elem == cur_elem )
      {
        add_elem = false;
        break;
      }
    }
    if( add_elem )
      gt_proj_nd_vtx_.push_back(new_elem);
  }

  cv::Mat gt_proj_nd_vtx_mat( gt_proj_nd_vtx_.size(), 2, CV_32FC1, gt_proj_nd_vtx_.data() );
  cv::Mat gt_proj_nd_vtx_id( gt_proj_nd_vtx_.size(), 1, CV_32FC1 );

  int *id_p = gt_proj_nd_vtx_id.ptr<int>(0);
  for( int i = 0; i < static_cast<int>(gt_proj_nd_vtx_.size()); i++, id_p++ )
    *id_p = static_cast<float>(i);

  knn_ = cv::ml::KNearest::create();
  knn_->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
  knn_->setDefaultK(1);
  knn_->train(gt_proj_nd_vtx_mat, cv::ml::ROW_SAMPLE, gt_proj_nd_vtx_id );
}

bool Projection2DMetric::performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, bool symmetric_object )
{
  std::vector<cv::Point2f> proj_vtx;

  obj_model_ptr_->setModelView( r_quat, t_vec );
  obj_model_ptr_->projectVertices( proj_vtx );

  double avg_pixel_diff = 0;

  if( symmetric_object )
  {
    cv::Mat proj_vtx_mat( proj_vtx.size(), 2, CV_32FC1, proj_vtx.data() );
    cv::Mat nn_proj_vtx_id( proj_vtx.size(), 1, CV_32SC1 );

    knn_->findNearest(proj_vtx_mat, 1, nn_proj_vtx_id );

    int *nn_vtx_id_p = nn_proj_vtx_id.ptr<int>(0);
    for (int i = 0; i < static_cast<int>(proj_vtx.size()); i++)
      avg_pixel_diff += cv_ext::norm2D( proj_vtx[i] - gt_proj_nd_vtx_[nn_vtx_id_p[i]] );
  }
  else
  {
    for (int i = 0; i < static_cast<int>(proj_vtx.size()); i++)
      avg_pixel_diff += cv_ext::norm2D(proj_vtx[i] - gt_proj_vtx_[i]);
  }

  avg_pixel_diff /= proj_vtx.size();
  return  ( avg_pixel_diff < max_proj_err_pix_ );
}

Pose6DMetric::Pose6DMetric( const RasterObjectModel3DPtr &obj_model_ptr, double obj_diameter_cm,
                            double max_err_percentage ) :
    obj_model_ptr_(obj_model_ptr),
    obj_diameter_cm_(obj_diameter_cm),
    max_err_percentage_(max_err_percentage)
{}

void Pose6DMetric::setGroundTruth(Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec)
{
  const std::vector<cv::Point3f> &model_vtx = obj_model_ptr_->vertices();
  gt_vtx_.resize(model_vtx.size() );

  cv::Point3f gt_t( gt_t_vec(0), gt_t_vec(1), gt_t_vec(2) );

  double tmp_gt_q[4];
  cv_ext::eigenQuat2Quat( gt_r_quat, tmp_gt_q );
  float gt_q[4];
  for( int i = 0; i < 4; i++ )
    gt_q[i] = static_cast<float>(tmp_gt_q[i]);

  for( int i = 0; i < static_cast<int>(model_vtx.size()); i++ )
  {
    ceres::UnitQuaternionRotatePoint( gt_q, reinterpret_cast<const float*>( &(model_vtx[i]) ),
                                      reinterpret_cast<float*>( &(gt_vtx_[i]) ) );

    gt_vtx_[i] += gt_t;
  }

  gt_nd_vtx_.clear();
  gt_nd_vtx_.reserve(gt_vtx_.size());

  // Remove duplicates
  bool add_elem;
  for( const auto &new_elem : gt_vtx_ )
  {
    add_elem = true;
    for( auto &cur_elem : gt_nd_vtx_ )
    {
      if( new_elem == cur_elem )
      {
        add_elem = false;
        break;
      }
    }
    if( add_elem )
      gt_nd_vtx_.push_back(new_elem);
  }

  cv::Mat gt_nd_vtx_mat( gt_nd_vtx_.size(), 3, CV_32FC1, gt_nd_vtx_.data() );
  cv::Mat gt_nd_vtx_id( gt_nd_vtx_.size(), 1, CV_32FC1 );

  int *id_p = gt_nd_vtx_id.ptr<int>(0);
  for( int i = 0; i < static_cast<int>(gt_nd_vtx_.size()); i++, id_p++ )
    *id_p = static_cast<float>(i);

  knn_ = cv::ml::KNearest::create();
  knn_->setAlgorithmType(cv::ml:: KNearest::BRUTE_FORCE);
  knn_->setDefaultK(1);
  knn_->train( gt_nd_vtx_mat, cv::ml::ROW_SAMPLE, gt_nd_vtx_id );
}

bool Pose6DMetric::performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, bool symmetric_object )
{
  const std::vector<cv::Point3f> &model_vtx = obj_model_ptr_->vertices();
  std::vector<cv::Point3f> vtx(model_vtx.size()), gt_vtx(model_vtx.size());

  cv::Point3f t( t_vec(0), t_vec(1), t_vec(2) );

  double tmp_q[4];
  cv_ext::eigenQuat2Quat( r_quat, tmp_q );
  float q[4];
  for( int i = 0; i < 4; i++ )
    q[i] = static_cast<float>(tmp_q[i]);

  for( int i = 0; i < static_cast<int>(model_vtx.size()); i++ )
  {
    ceres::UnitQuaternionRotatePoint( q, reinterpret_cast<const float*>( &(model_vtx[i]) ),
                                      reinterpret_cast<float*>( &(vtx[i]) ) );
    vtx[i] += t;
  }

  double avg_diff = 0;
  if( symmetric_object )
  {
    cv::Mat vtx_mat( vtx.size(), 3, CV_32FC1, vtx.data() );
    cv::Mat nn_vtx_id( vtx.size(), 1, CV_32SC1 );

    knn_->findNearest( vtx_mat, 1, nn_vtx_id );

    int *nn_vtx_id_p = nn_vtx_id.ptr<int>(0);
    for( int i = 0; i < static_cast<float>(vtx.size()); i++ )
      avg_diff += cv_ext::norm3D( vtx[i] - gt_nd_vtx_[nn_vtx_id_p[i]] );
  }
  else
  {
    for( int i = 0; i < static_cast<int>(vtx.size()); i++ )
      avg_diff += cv_ext::norm3D( vtx[i] - gt_vtx_[i] );
  }

  avg_diff /= vtx.size();

  return  ( 100.0*avg_diff < max_err_percentage_*obj_diameter_cm_ );
}
