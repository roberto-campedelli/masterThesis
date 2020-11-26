/*
 * cv_ext - openCV EXTensions
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */


#pragma once

#include <string>
#include <vector>
#include <utility>
#include <list>

#include "cv_ext/pinhole_camera_model.h"
#include "cv_ext/stereo_camera.h"

/* TODO
 *
 *  -MultiStereoCameraCalibration : check connections between cameras. Connected graph?
 */

namespace cv_ext
{

/** @brief Abstract base class for calibration objects */       
class CameraCalibrationBase
{
public:

  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] cache_folder Path of the directory used to cache the calibration images
   * 
   * The constructor creates the directory cache_folder used to temporarily cache the calibration images. 
   * If this directory exits, the constructor deletes all the images previously saved
   */
  CameraCalibrationBase( int cache_max_size, const std::string &cache_folder );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~CameraCalibrationBase() = 0;
  
  /** @brief Set the size of the board, i.e. the number of internal corners by width and height
   *
   * @warning Horizontal and vertical size must differ
   * */
  void setBoardSize( const cv::Size &s );
  
  /** @brief Provide the size of the board, i.e. the number of internal corners by width and height */
  cv::Size boardSize(){ return board_size_; };
  
  /** @brief Set the size of a checkerboard square in the defined unit (points, millimeters, etc) */
  void setSquareSize( float s ){ square_size_ = s; };
  
  /** @brief Provide the size of a checkerboard square in the defined unit (points, millimeters, etc) */
  float squareSize(){ return square_size_; };
  
  /** @brief If the checkerboard has a white circle in the black square corresponding to the origin 
   *         of the reference frame, call this method with the enable parameter as true
   * 
   * Set to false by default
   *
   * @note The use of a checkerboard with a white circle in the black square corresponding to the origin 
   *       of the reference frame is generally recommended
   */
  void setUseCheckerboardWhiteCircle( bool enable ){ pattern_has_white_circle_ = enable; };

  /** @brief Return true if the checkerboard has a white circle in the black square corresponding to the origin
   *         of the reference frame
   */
  bool useCheckerboardWhiteCircle(){ return pattern_has_white_circle_; };

  /** @brief Set the the maximum number of iterations of the calibration algorithm */
  void setMaxNumIter( int num ){ max_iter_ = num; };

  /** @brief Provide the the maximum number of iterations of the calibration algorithm */
  int maxNumIter(){ return max_iter_; };
  
  /** @brief Set the number of levels of the gaussian image pyramid used to extract corners
   * 
   * @param[in] pyr_num_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * 
   * Build a gaussian pyramid of the image and start to extract the corners from the higher
   * level of a gaussian pyramid. For large images, it may be useful to use a 2 or 3 levels pyramid.
   * Set to 1 by default
   */
  void setPyramidNumLevels( int pyr_num_levels ) { pyr_num_levels_ = pyr_num_levels; };

  /** @brief Provide the number of levels of the gaussian image pyramid used to extract corners, 
   *         see setPyramidNumLevels() */
  int pyramidNumLevels() { return pyr_num_levels_; };

  /** @brief Provide the size of the images used for calibration
   * 
   * A (-1,-1) size is returnend if no images have been added
   */  
  cv::Size imagesSize(){ return image_size_; }
  
  /** @brief Provide the OpenCV image type of the images used for calibration
   * 
   * See cv::Mat::type()
   * A -1 type is returnend if no images have been added
   */  
  int imagesType(){ return image_type_; }

  /** @brief Pure virtual method that should be overridden in derived classes
   * 
   * It should run the calibration given the loadaed/selected images.
   * 
   * @return Some metric about the calibration
   */
  virtual double calibrate() = 0;
  
  /** @brief Pure virtual method that should be overridden in derived classes.
   *
   *  It should clear all the loaded images and the chache 
   **/
  virtual void clear() = 0;
  
protected:
  
  void clearDiskCache();
  void cachePutImage( int i, const cv::Mat &img );
  cv::Mat cacheGetImage( int i );
  std::string getCacheFilename( int i );

  cv::Size board_size_ = cv::Size(-1,-1);
  float square_size_ = 0.0f;
  bool pattern_has_white_circle_ = false;
  int max_iter_ = 30;
  int pyr_num_levels_ = 1;
  
  cv::Size image_size_ = cv::Size(-1,-1);
  int image_type_;

private:
  /** List used to implement a very simple, linear-time access cache */
  std::list< std::pair<int, cv::Mat> > images_cache_;

  int cache_max_size_;
  std::string cache_folder_;
};



/** @brief Calibrate a single camera
 *
 *  CameraCalibration estimates the intrinsic parameters (both K and distortion parameters) of a cameras.
 **/
class CameraCalibration : public CameraCalibrationBase
{
public:

  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] cache_folder Path of the directory used to cache the calibration images
   * 
   * The constructor creates the directory cache_folder used to temporarily cache the calibration images. 
   * If this directory exits, the constructor deletes all the images previously saved
   */
  explicit CameraCalibration( int cache_max_size = 100, const std::string &cache_folder = "/tmp" );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~CameraCalibration() = default;

  /** @brief If enabled, the calibration assumes zero tangential distortion
   *
   * Set to false by default
   */
  void setZeroTanDist( bool enable ){ zero_tan_dist_ = enable; };

  /** @brief Return true if the calibration assumes zero tangential distortion  */
  bool zeroTanDist(){ return zero_tan_dist_; };

  /** @brief If enabled, the calibration uses a provided camera model as initial guess
 *
 * Set to false by default
 */
  void setUseIntrinsicGuess( bool enable ){ use_intrinsic_guess_ = enable; };

  /** @brief Return true if the calibration uses a provided camera model as initial guess
   */
  bool useIntrinsicGuess(){ return use_intrinsic_guess_; };

  /** @brief If enabled, consider in the calibration only fy as a free parameter, with fx/fy = 1
   *
   * Set to false by default
   */
  void setFixAspectRatio( bool enable ){ fix_aspect_ratio_ = enable; };

  /** @brief Return true if the calibration considers only fy as a free parameter, with fx/fy = 1
   */
  bool fixAspectRatio(){ return fix_aspect_ratio_; };

  /** @brief If enabled, the principal point is not changed during the global optimization
   *
   * Set to false by default
   */
  void setFixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };

  /** @brief Return true if the principal point is not changed during the global optimization
   */
  void fixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };

  /** @brief Set a previously computed camera model
   * 
   * @param[in] cam_model Input camera model
   * 
   * If setUseIntrinsicGuess() is set to true, this model will be used as an initial guess in the calibration
   */
  void setCamModel( const PinholeCameraModel &model );
  
  /** @brief Provide the resulting camera model 
   *
   * @return The estimated model
   *
   * @note If no calibration has been performed, or no models have been set with setCamModel(),
   *       this method provides a default PinholeCameraModel object
   */
  PinholeCameraModel getCamModel();
  
  /** Add an image of a checkerboard.
   * 
   * @param[in] img A reference to a one or three channels image
   *
   * @return True if the image is valid and the checkerboard has been
   *         succesfully extracted, false otherwise
   */
  bool addImage( const cv::Mat &img );

  /** Add an image of a checkerboard and the pre-extracted corners.
   *
   * @param[in] img A reference to a one or three channels image
   * @param[in] corner_pts A vector of corner points extracted from the image
   *
   */
  void addImage( const cv::Mat &img, const std::vector<cv::Point2f> &corners_pts );

  /** @brief Load and add an image of a checkerboard.
   * 
   * @param[in] filename Path of image file to be loaded.
   * 
   * @return True if the image is valid and the checkerboard has been 
   *         succesfully extracted, false otherwise
   */  
  bool addImageFile ( const std::string &filename );

  /** @brief Provide the number of images succesfully added with the addImage() or addImageFile()
   *         methods so far */
  int numImages(){ return num_images_; };

  /** @brief Provide the number of active images, i.e. the images actually used for 
   *         calibration (see setImageActive() and calibrate()) */
  int numActiveImages(){ return num_active_images_; };
  
  /** @brief Set whether an added image will be used in the calibration process
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * @param[in] active If false, the image with index i will not be used for calibration
   * 
   * By default, all added images are used for calibration.
   */
  void setImageActive( int i, bool active );
  
  /** Return true if an added image will be used in the calibration process
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   */
  bool isImageActive( int i ){ return images_masks_[i]; };
  
  /** @brief Perform the calibration.
   * 
   * @return The root mean squared re-projection error, computed only on the images marked as active
   * 
   * The calibration is performed using only the images marked as active (see setImageActive())
   * If no images have been added (see addImage() and addImageFile()), or all images have been marked 
   * as not active, this method will return an infinite value
   */
  double calibrate() override;

  /** @brief Provide a root mean squared reprojection error for the i-th image 
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * 
   * If the last calibration has not been performed using the required image, 
   * this method will return an infinite value
   */
  double getReprojectionError( int i ){ return per_view_errors_[i]; };

  /** @brief Provides the extracted corners for the ith image
   *
   * @param[in] i Index of the image, from 0 to numImages() - 1
   *
   * @return Vector with the extracted corners
   */
  std::vector<cv::Point2f> getCorners( int i );

  /** @brief Provide a possibly scaled image with drawn the detected checkerboard corners
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * @param[out] corners_img Output image with represented the extracted corners
   * @param[in] scale_factor The output image scale factor, it should be >= 1 
   */  
  void getCornersImage( int i, cv::Mat &corners_img, float scale_factor = 1.0f );

  /** @brief Provide a possibly scaled image that depicts a qualitative representation of the 
   *  non-normalized density of the checkerboard corners
   * 
   * @param[in] kernel_stdev Standard deviation of the Gaussian kernel used in the density 
   *                         estimation
   * @param[out] corner_dist Output corner distribution image  (one channel, depth CV_32F)
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   * 
   * getCornersDistribution() considers the corners extracted from each added image (see addImage() and 
   * addImageFile()) marked as active (see setImageActive()).
   * The density is obtained by means of kernel density estimation, i.e. a simplified version 
   * of the Parzen–Rosenblatt window method, using a non-normalized Gaussian kernel (kernel(0,0) = 1) 
   * with standard deviation kernel_stdev.
   */  
  void getCornersDistribution( float kernel_stdev, cv::Mat &corner_dist, float scale_factor = 1.0f );
  
  /** @brief Provide a possibly scaled image, undistorted using the current calibration parameters
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * @param[out] und_img Output undistorted image
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   */
  void getUndistortedImage( int i, cv::Mat &und_img, float scale_factor = 1.0f );

  /** @brief Clear all the loaded images and the chache **/
  void clear();

  void computeAverageExtrinsicParameters( cv::Mat &r_vec, cv::Mat &t_vec );
  // TODO Move this method away??

private:

  bool use_intrinsic_guess_ = false,
       fix_aspect_ratio_ = false,
       zero_tan_dist_ = false,
       fix_principal_point_ = false;

  int num_images_ = 0, num_active_images_ = 0;
  cv::Mat camera_matrix_, dist_coeffs_;
  
  std::vector< std::vector<cv::Point2f> > images_points_;
  std::vector<bool> images_masks_;
  std::vector<double> per_view_errors_;
};

/** @brief Calibrate a stereo pair
 *
 *  StereoCameraCalibration can estimates the extrinsics and (if required) also the intrinsic parameters of
 *  the two cameras (both K and distortion parameters).
 **/
class StereoCameraCalibration : public CameraCalibrationBase
{ 
public:
  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] cache_folder Path of the directory used to cache the calibration images
   * 
   * The constructor creates the directory cache_folder used to temporarily cache the calibration images. 
   * If this directory exits, the constructor deletes all the images previously saved
   */
  explicit StereoCameraCalibration( int cache_max_size = 100, const std::string &cache_folder = "/tmp" );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~StereoCameraCalibration() = default;

  /** @brief If enabled, the calibration assumes zero tangential distortion
   *
   * Set to false by default
   */
  void setZeroTanDist( bool enable ){ zero_tan_dist_ = enable; };

  /** @brief Return true if the calibration assumes zero tangential distortion  */
  bool zeroTanDist(){ return zero_tan_dist_; };

  /** @brief If enabled, the calibration uses a provided camera model as initial guess
 *
 * Set to false by default
 */
  void setUseIntrinsicGuess( bool enable ){ use_intrinsic_guess_ = enable; };

  /** @brief Return true if the calibration uses a provided camera model as initial guess
   */
  bool useIntrinsicGuess(){ return use_intrinsic_guess_; };

  /** @brief If enabled, consider in the calibration only fy as a free parameter, with fx/fy = 1
   *
   * Set to false by default
   */
  void setFixAspectRatio( bool enable ){ fix_aspect_ratio_ = enable; };

  /** @brief Return true if the calibration considers only fy as a free parameter, with fx/fy = 1
   */
  bool fixAspectRatio(){ return fix_aspect_ratio_; };

  /** @brief If enabled, the principal point is not changed during the global optimization
   *
   * Set to false by default
   */
  void setFixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };

  /** @brief Return true if the principal point is not changed during the global optimization
   */
  void fixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };

  /** @brief If enabled, enforce the focal lenghts to be the same for both camera
   *
   * This method has effect if no camera model has been provided in input (see setCamModels()) or
   * setUseIntrinsicGuess() is set to true.
   * Set to false by default
   */
  void setForceSameFocalLenght( bool enable ){ force_same_focal_lenght_ = enable; }; 

  /** @brief Return true if the calibration enforces the focal lenghts to be the same for both camera, 
   *         see setForceSameFocalLenght() */
  bool forceSameFocalLenght(){ return force_same_focal_lenght_; }; 
  
  /** @brief Set previously computed camera models
   * 
   * @param[in] cam_models Input vector of two camera models
   * 
   * Before calibrate a stereo camera, it is recommended to calibrate the cameras 
   * individually using CameraCalibration an to provide the resulting camera parameters to 
   * StereoCameraCalibration using setCamModels().
   * If setUseIntrinsicGuess() is set to true, this models will be used as an initial guess in 
   * the calibration, otherwise these parameters will be keep fixed and the calibration will 
   * estimate only the extrinsics parameters between the cameras.
   */  
  void setCamModels( const std::vector < PinholeCameraModel > &cam_models );
 
  /** @brief Provide the resulting camera models
   *
   * @return A vector with the estimated camera models
   * 
   * @note If no calibration has been performed, or no models have been set with setCamModels(),
   *       this method provides two default PinholeCameraModel objects.
   */
  std::vector < PinholeCameraModel > getCamModels();
  
  /** @brief Provide the resulting extrinsic parameters
   * 
   * @param[out] r_mat Rotation matrix between the first and the second camera
   * @param[out] t_vec Translation vector between the two cameras
   * 
   * @note If no calibration has been performed, this method provides two empty matrices
   */
  void getExtrinsicsParameters( cv::Mat &r_mat, cv::Mat &t_vec );
  
  /** Add a pair of images of a checkerboard acquired at the same time by the stereo pair
   * 
   * @param[in] imgs A vector of two one or three channels images.
   * 
   * @return True if the images are valid and the checkerboard has been 
   *         succesfully extracted in both images, false otherwise.
   *
   * Both camera should frame the checkerboard.
   */  
  bool addImagePair( const std::vector< cv::Mat > &imgs );


  /** Add a pair of images of a checkerboard acquired at the same time by the stereo pair, and the pre-extracted corners.
   *
   * @param[in] imgs A vector of two one or three channels images.
   * @param[in] corner_pts A vector of two vectors of corner points extracted from the image pair
   *
   */
  void addImagePair( const std::vector< cv::Mat > &imgs, const std::vector< std::vector<cv::Point2f> > &corners_pts );

  /** @brief Load and add a pair of images of a checkerboard acquired at the same time by the stereo pair
   * 
   * @param[in] filenames A vector with the paths of the two image files to be loaded.
   * 
   * @return True if the images are valid and the checkerboard has been 
   *         succesfully extracted in both images, false otherwise
   *
   * Both camera should frame the checkerboard.
   */   
  bool addImagePairFiles ( const std::vector< std::string > &filenames );

  /** @brief Provide the number of image pairs succesfully added with the addImagePair() or
   *         addImagePairFiles() methods so far */  
  int numImagePairs(){ return num_pairs_; };

  /** @brief Provide the number of active image pairs, i.e. the pairs actually used for 
   *         calibration (see setImagePairActive() and calibrate()) */  
  int numActiveImagePairs(){ return num_active_pairs_; };
  
  /** @brief Set whether an added image pair will be used in the calibration process
   * 
   * @param[in] i Index of the pair, from 0 to numImagePairs() - 1
   * @param[in] active If false, the image pair with index i will not be used for calibration
   * 
   * By default, all added image pairs are used for calibration.
   */  
  void setImagePairActive( int i, bool active );
  
  /** Return true if an added image pair will be used in the calibration process
   * 
   * @param[in] i Index of the image, from 0 to numImagePairs() - 1
   */  
  bool isImagePairActive( int i ){ return pairs_masks_[i]; };
  
  /** @brief Perform the calibration.
   * 
   * @return The root mean squared (RMS) distance between the extracted points and estimated epipolar lines.
   *
   * The calibration is performed using only the image pairs marked as active (see setImagePairActive())
   * If no image pairs have been added (see addImagePair() and addImagePairFiles()), or all images have been marked 
   * as not active, this method will return an infinite value.
   * The returned RMS error is computed using the epipolar geometry constraint: m2^t*F*m1=0, where the fundamental
   * matrix F is computed using the estimated relative transformation between cameras.
   * @note The RMSE is computed conisdering  only on the image pairs marked as active.
   */  
  double calibrate() override;

  /** @brief Provide the RMS distance between the extracted points and estimated epipolar lines for the i-th image pair
   * 
   * @param[in] i Index of the image pair, from 0 to numImagePairs() - 1
   * 
   * If the last calibration has not been performed using the required image pair, 
   * this method will return an infinite value
   */
  double getEpipolarError( int i ){ return per_view_errors_[i]; };

  /** @brief Provides the extracted corners for the ith image pair
   *
   * @param[in] i Index of the image pair, from 0 to numImagePairs() - 1
   *
   * @return Vectors of vector with the extracted corners
   */
  std::vector< std::vector< cv::Point2f > > getCornersPair( int i );

  /** @brief Provide a pair of possibly scaled images with drawn the detected checkerboard corners
   * 
   * @param[in] i Index of the image pair, from 0 to numImagePairs() - 1
   * @param[out] corners_img Output image pair with represented the extracted corners
   * @param[in] scale_factor The output image scale factor, it should be >= 1 
   */  
  void getCornersImagePair( int i, std::vector< cv::Mat > &corners_imgs, float scale_factor = 1.0f );
  
  /** @brief Provide a pair of possibly scaled images that depicts a qualitative representation of the
   *  non-normalized density of the checkerboard corners
   * 
   * @param[in] kernel_stdev Standard deviation of the Gaussian kernel used in the density 
   *                         estimation
   * @param[out] corner_dists Output corner distribution images (one channel, depth CV_32F)
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   * 
   * getCornersDistribution() considers the corners extracted from each added image pair
   * (see addImagePair() and addImagePairFiles()) marked as active (see setImagePairActive()).
   * The density is obtained by means of kernel density estimation, i.e. a simplified version 
   * of the Parzen–Rosenblatt window method, using a non-normalized Gaussian kernel (kernel(0,0) = 1) 
   * with standard deviation kernel_stdev
   */  
  void getCornersDistribution( float kernel_stdev, std::vector< cv::Mat > &corner_dists, float scale_factor = 1.0f );
  
  /** @brief Provide a pair of possibly scaled rectified images
   * 
   * @param[in] i Index of the image pair to be rectified, from 0 to numImagePairs() - 1
   * @param[out] corners_img Output rectified image pair
   * @param[in] scale_factor The output image scale factor
   *
   * This method internally uses the StereoRectification() object, can can be called only after calibrate(),
   * otherwise two empty images will be returned.
   */  
  void rectifyImagePair( int i, std::vector< cv::Mat > &rect_imgs, float scale_factor = 1.0f );

  /** @brief Clear all the loaded images and the chache **/
  void clear();
  
private:
  
  int num_pairs_ = 0, num_active_pairs_ = 0;

  bool use_intrinsic_guess_ = false,
       fix_aspect_ratio_ = false,
       zero_tan_dist_ = false,
       fix_principal_point_ = false,
       force_same_focal_lenght_ = false;

  cv::Mat camera_matrices_[2], dist_coeffs_[2];
  cv::Mat r_mat_, t_vec_;
  
  std::vector< std::vector<cv::Point2f> > images_points_[2];
  std::vector<bool> pairs_masks_;
  std::vector<double> per_view_errors_;
  
  cv_ext::StereoRectification stereo_rect_;
};

/** @brief Calibrate a multi-stereo system (i.e., a N-cameras rig)
 *
 *  MultiStereoCameraCalibration only estimates the extrinsics parameters of the cameras (i.e., the rigid body
 *  transformations between the first camera and each other camera in the rig).
 */
class MultiStereoCameraCalibration : public CameraCalibrationBase
{
 public:

  /** @brief Object constructor
   *
   * @param[in] num_cameras Number of cameras included in the rig
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] cache_folder Path of the directory used to cache the calibration images
   *
   * The constructor creates the directory cache_folder used to temporarily cache the calibration images.
   * If this directory exits, the constructor deletes all the images previously saved
   */
  explicit MultiStereoCameraCalibration( int num_cameras, int cache_max_size = 100, const std::string cache_folder = "/tmp" );

  /** @brief Object destructor
   *
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~MultiStereoCameraCalibration() = default;

  /** @brief Set the previously computed camera models, on for each camera of the rig
   *
   * @param[in] cam_models Input vector of camera models
   *
   * MultiStereoCameraCalibration only estimates the extrinsics parameters between the cameras.
   * The intrinsic parameters of each camera (both K and distortion parameters) should be estimated
   * in advance, e.g. by using the CameraCalibration object.
   */
  void setCamModels( const std::vector< PinholeCameraModel > &cam_models );

  /** @brief Set the alpha parameter used in the Huber loss function during the final calibration refinement */
  void setHuberLossAlpha( double alpha ){ huber_loss_alpha_ = alpha; }

  /** @brief Provide the alpha parameter used in the Huber loss function during the final calibration refinement */
  double huberLossAlpha(){ return huber_loss_alpha_; }

  /** @brief Provide the resulting extrinsic parameters
   *
   * @param[out] r_mats Vector of the rotation matrices between the first and the other cameras
   * @param[out] t_vecs Vector of the translation vectors between the first and the other cameras
   *
   * The extinisc parameters of a N-cameras rig are represented by the N-1 rigid body transformations between the
   * first camera and the other N-1 cameras in the rig.
   * @note If no calibration has been performed, this method provides empty vectors.
   */
  void getExtrinsicsParameters( std::vector< cv::Mat > &r_mats, std::vector< cv::Mat > &t_vecs );

  /** Add a tuple of images of a checkerboard acquired at the same time by the N-cameras rig.
   *
   * @param[in] imgs A vector of one or three channels images
   *
   * @return True if the images are valid and the checkerboard has been
   *         succesfully extracted in at least two images, false otherwise
   *
   * At least two cameras should frame the checkerboard.
   */
  bool addImageTuple(const std::vector<cv::Mat> &imgs );

  void addImageTuple(const std::vector<cv::Mat > &imgs, const std::vector<std::vector<cv::Point2f> > &corners_pts );

  /** @brief Load and add a tuple of images of a checkerboard acquired at the same time by the N-cameras rig.
   *
   * @param[in] filenames A vector with the paths of image files to be loaded.
   *
   * @return True if the images are valid and the checkerboard has been
   *         succesfully extracted in at least two images, false otherwise.
   *
   * At least two cameras should frame the checkerboard.
   */
  bool addImageTupleFiles ( const std::vector< std::string > &filenames );

  /** @brief Provide the number of image pairs succesfully added with the addImagePair() or
   *         addImagePairFiles() methods so far */
  int numImageTuples(){ return num_tuples_; };

  /** @brief Provide the number of active image pairs, i.e. the pairs actually used for
   *         calibration (see setImagePairActive() and calibrate()) */
  int numActiveImageTuples(){ return num_active_tuples_; };

  /** @brief Set whether an added image pair will be used in the calibration process
   *
   * @param[in] i Index of the pair, from 0 to numImagePairs() - 1
   * @param[in] active If false, the image pair with index i will not be used for calibration
   *
   * By default, all added image pairs are used for calibration.
   */
  void setImageTupleActive( int i, bool active );

  /** Return true if an added image pair will be used in the calibration process
   *
   * @param[in] i Index of the image, from 0 to numImagePairs() - 1
   */
  bool isImageTupleActive( int i ){ return tuples_masks_[i]; };

  /** @brief Perform the calibration.
   *
   * @return The root mean squared (RMS) distance between the extracted points and estimated epipolar lines.
   *
   * The calibration is performed using only the image tuples marked as active (TODO see setImagePairActive())
   * If no image tuples have been added (see addImageTuple() and addImageTupleFiles()), or all tuples have been marked
   * as not active, this method will return an infinite value.
   * The returned RMS error is computed using the epipolar geometry constraint: m2^t*F*m1=0, where the fundamental
   * matrices F are computed using the estimated relative transformations between cameras.
   * @note The RMSE is computed conisdering  only on the image tuples marked as active.
   */
  double calibrate() override;

  /** @brief Provide the RMS distance between the extracted points and estimated epipolar lines for the i-th image tuple
   *
   * @param[in] i Index of the image tuple, from 0 to numImageTuples() - 1
   *
   * If the last calibration has not been performed using the required image tuple,
   * this method will return an infinite value
   */
  double getEpipolarError( int i ){ return per_view_errors_[i]; };

  /** @brief Provide a tuple possibly scaled images with drawn the detected checkerboard corners
   *
   * @param[in] i Index of the image pair, from 0 to numImagePairs() - 1
   * @param[out] corners_img Output image pair with represented the extracted corners
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   */
  void getCornersImageTuple( int i, std::vector< cv::Mat > &corners_imgs, float scale_factor = 1.0f );

  /** @brief Provide a tuple of possibly scaled images that depicts a qualitative representation of the
   *  non-normalized density of the checkerboard corners
   *
   * @param[in] kernel_stdev Standard deviation of the Gaussian kernel used in the density
   *                         estimation
   * @param[out] corner_dists Output corner distribution images (one channel, depth CV_32F)
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   *
   * getCornersDistribution() considers the corners extracted from each added image tuple
   * (see addImageTuple() and addImageTupleFiles()) marked as active (see setImageTupleActive()).
   * The density is obtained by means of kernel density estimation, i.e. a simplified version
   * of the Parzen–Rosenblatt window method, using a non-normalized Gaussian kernel (kernel(0,0) = 1)
   * with standard deviation kernel_stdev
   */
  void getCornersDistribution( float kernel_stdev, std::vector< cv::Mat > &corner_dists, float scale_factor = 1.0f );

  /** @brief Clear all the loaded images and the chache **/
  void clear();

 private:

  int num_cameras_;
  int num_tuples_ = 0, num_active_tuples_ = 0;
  std::vector< cv_ext::PinholeCameraModel > cam_models_;

  std::vector< cv::Mat > r_mats_, t_vecs_;
  double huber_loss_alpha_ = 1.0;

  std::vector< std::vector< std::vector<cv::Point2f> > > images_points_;
  std::vector<bool> tuples_masks_;
  std::vector<double> per_view_errors_;
};

}
