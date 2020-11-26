#pragma once

#include "raster_object_model3D.h"

class EvaluationMetricBase
{
 public:

  virtual void setGroundTruth( Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec ) = 0;
  virtual bool performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, bool symmetric_object ) = 0;
};

class MaxRtErrorsMetric : public EvaluationMetricBase
{
 public:

  MaxRtErrorsMetric( double max_rot_err_deg = 5., double max_t_err_cm = 5. );

  double maxRotationErrorDeg() const { return max_rot_err_deg_; };
  double maxTranslationErrorCm() const { return max_t_err_cm_; };
  void setMaxRotationErrorDeg( double error ){ max_rot_err_deg_ = error; };
  void setMaxTranslationErrorCm( double error ){ max_t_err_cm_ = error; };

  void setGroundTruth( Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec ) override;
  bool performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, bool symmetric_object = false ) override;

 private:

  double max_rot_err_deg_, max_t_err_cm_;

  Eigen::Quaterniond gt_r_quat_;
  Eigen::Vector3d gt_t_vec_;
};

class Projection2DMetric : public EvaluationMetricBase
{
 public:

  Projection2DMetric( const RasterObjectModel3DPtr &obj_model_ptr, double max_proj_err_pix = 5. );

  RasterObjectModel3DPtr objectModel(){ return obj_model_ptr_; };
  void setObjectModel( const RasterObjectModel3DPtr &obj_model_ptr ){ obj_model_ptr_ = obj_model_ptr; };

  double maxProjectionErrorPix() const { return max_proj_err_pix_; };
  void setMaxProjectionErrorPix( double error ){ max_proj_err_pix_ = error; };

  void setGroundTruth( Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec ) override;
  bool performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, bool symmetric_object = false  ) override;

 private:

  RasterObjectModel3DPtr obj_model_ptr_;
  double max_proj_err_pix_;

  std::vector<cv::Point2f> gt_proj_vtx_, gt_proj_nd_vtx_;
  cv::Ptr<cv::ml::KNearest> knn_;
};

class Pose6DMetric : public EvaluationMetricBase
{
 public:

  Pose6DMetric(const RasterObjectModel3DPtr &obj_model_ptr, double obj_diameter_cm,
               double max_err_percentage = .1 );

  RasterObjectModel3DPtr objectModel(){ return obj_model_ptr_; };
  void setObjectModel( const RasterObjectModel3DPtr &obj_model_ptr ){ obj_model_ptr_ = obj_model_ptr; };

  double objectDiameterCm() const { return obj_diameter_cm_; };
  void setObjectDiameterCm( double diameter ){ obj_diameter_cm_ = diameter; };
  double maxErrorPercentage() const { return max_err_percentage_; };
  void setMaxErrorPercentage( double percentage ){ max_err_percentage_ = percentage; };

  void setGroundTruth( Eigen::Quaterniond &gt_r_quat, Eigen::Vector3d &gt_t_vec ) override;
  bool performTest( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec, bool symmetric_object = false ) override;

 private:

  RasterObjectModel3DPtr obj_model_ptr_;
  double obj_diameter_cm_, max_err_percentage_;

  std::vector<cv::Point3f> gt_vtx_, gt_nd_vtx_;
  cv::Ptr<cv::ml::KNearest> knn_;
};
