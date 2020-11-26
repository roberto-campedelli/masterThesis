#include "cv_ext/camera_calibration.h"
#include "cv_ext/macros.h"
#include "cv_ext/image_pyramid.h"
#include "cv_ext/conversions.h"
#include "cv_ext/debug_tools.h"
#include "cv_ext/pinhole_scene_projector.h"
#include "cv_ext/calibration_cost_functions.h"

#include <ceres/rotation.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <uuid/uuid.h>

#include <algorithm>

using namespace cv;
using namespace std;
using namespace cv_ext;
using namespace boost::filesystem;

static const std::string cache_folder_basename ( ".camera_calibration_cache_" );
static const std::string cache_file_basename ( "camera_calibration_file_" );

static double getMedian( std::vector<double > vals )
{
  size_t size = vals.size();

  if (size == 0)
  {
    return 0;
  }
  else
  {
    std::sort(vals.begin(), vals.end());
    if (size % 2 == 0)
      return (vals[size / 2 - 1] + vals[size / 2]) / 2;
    else
      return vals[size / 2];
  }
}

static vector<Point3f> getBoardCornerPositions (Size board_size, float square_size )
{
  vector<Point3f> object_points;
  object_points.reserve(board_size.width*board_size.height);
  for ( int i = 0; i < board_size.height; ++i )
  {
    for ( int j = 0; j < board_size.width; ++j )
    {
      object_points.push_back ( Point3f ( float ( j*square_size ), float ( i*square_size ), 0 ) );
    }
  }
  return object_points;
}

static bool findCheckerboardCornersPyr ( Mat img, Size board_size, vector<Point2f> &img_pts,
                                       int pyr_num_levels, bool has_white_circle )
{
  bool corners_found = false;
  ImagePyramidBase<cv_ext::MEM_ALIGN_NONE> img_pyr( img, pyr_num_levels, -1, false );

  if ( findChessboardCorners ( img_pyr[pyr_num_levels - 1], board_size, img_pts,
                               CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE  ) )
  {
    if( static_cast<int>( img_pts.size() ) == board_size.width*board_size.height )
    {
      corners_found = true;

      if ( has_white_circle )
      {
        // Look for the white circle in the pattern corresponding to the origin of the reference frame
        vector<Point2f> square_pts ( 4 ), square_img_pts ( 4 ), test_pts ( 5 ), test_img_pts ( 5 );

        square_pts[0] = Point2f ( 0.0f, 0.0f );
        square_pts[1] = Point2f ( 1.0f, 0.0f );
        square_pts[2] = Point2f ( 0.0f, 1.0f );
        square_pts[3] = Point2f ( 1.0f, 1.0f );

        square_img_pts[0] = img_pts[0];
        square_img_pts[1] = img_pts[1];
        square_img_pts[2] = img_pts[board_size.width];
        square_img_pts[3] = img_pts[board_size.width + 1];

        Mat h1 = getPerspectiveTransform ( square_pts, square_img_pts );

        int n_pts = img_pts.size();
        square_img_pts[0] = img_pts[n_pts - board_size.width - 2];
        square_img_pts[1] = img_pts[n_pts - board_size.width - 1];
        square_img_pts[2] = img_pts[n_pts - 2];
        square_img_pts[3] = img_pts[n_pts - 1];

        Mat h2 = getPerspectiveTransform ( square_pts, square_img_pts );

        test_pts[0] = Point2f ( 0.5f, 0.5f );
        test_pts[1] = Point2f ( 0.6f, 0.5f );
        test_pts[2] = Point2f ( 0.4f, 0.5f );
        test_pts[3] = Point2f ( 0.5f, 0.6f );
        test_pts[4] = Point2f ( 0.5f, 0.4f );

        perspectiveTransform ( test_pts, test_img_pts, h1 );

        float score1 = 0.0f, score2 = 0.0f;

        for ( auto &p : test_img_pts )
        {
          score1 += img_pyr[pyr_num_levels - 1].at<uchar> ( p.y, p.x );
        }

        perspectiveTransform ( test_pts, test_img_pts, h2 );

        for ( auto &p : test_img_pts )
        {
          score2 += img_pyr[pyr_num_levels - 1].at<uchar> ( p.y, p.x );
        }

        if ( score2 > score1 )
        {
          std::reverse ( img_pts.begin(),img_pts.end() );
        }

      }
      cornerSubPix ( img_pyr[pyr_num_levels - 1], img_pts, Size ( 11,11 ),
                     Size ( -1,-1 ), TermCriteria ( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ) );

      for ( int l = pyr_num_levels - 2; l >= 0; l-- )
      {
        for ( auto &p : img_pts )
        {
          p *= 2.0f;
        }
        cornerSubPix ( img_pyr[l], img_pts, Size ( 11,11 ),
                       Size ( -1,-1 ), TermCriteria ( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ) );
      }
    }
  }
  if( !corners_found )
    img_pts.clear();

  return corners_found;
}

static std::vector< std::vector <int > > bellmanFordShortestPath(const cv::Mat_<double> &adj_mat, int source_node)
{
  auto s = adj_mat.size();
  int dim = std::min(s.width, s.height);

  cv::Mat_<double> cost(dim,dim);
  for( int i=0; i<dim; i++ )
  {
    for( int j=0; j<dim; j++ )
    {
      if( adj_mat(i,j) == 0 )
        cost(i,j) = static_cast<double>(std::numeric_limits<int>::max());
      else
        cost(i,j) = adj_mat(i,j);
    }
  }

  std::vector <double > dist(dim);
  std::vector <int> prev(dim);
  for( int i = 0; i < dim; i++ )
  {
    dist[i] = static_cast<double>(std::numeric_limits<int>::max());
    prev[i] = -1;
  }

  dist[source_node] = 0;
  double new_dist;
  // After dim - 1 iteration, if no negative cycles exists, we are done
  for( int iter = 0; iter < dim - 1; iter++ )
  {
    for( int i=0; i<dim; i++ )
    {
      for( int j=0; j<dim; j++ )
      {
        new_dist = dist[i] + cost(i,j);
        if( new_dist < dist[j] )
        {
          dist[j] = new_dist;
          prev[j] = i;
        }
      }
    }
  }

//  std::cout<<"dist = [";
//  for(int i=0; i < dim; i++) std::cout<<dist[i]<<", ";
//  std::cout<<"]"<<std::endl;
//  std::cout<<"prev = [";
//  for(int i=0; i < dim; i++) std::cout<<prev[i]<<", ";
//  std::cout<<"]"<<std::endl;

  std::vector< std::vector <int> > paths(dim);
  for(int i=0, j; i < dim; i++)
  {
    if( i!=source_node )
    {
      j=i;
      paths[i].push_back(j);
      do
      {
        j=prev[j];
        paths[i].push_back(j);
      }while(j != source_node);
    }
  }
  return paths;
}


static inline double computeEpipolarSquareDistance ( const Eigen::Matrix3d &ess_mat,
                                                     const cv::Point2f &np0,
                                                     const cv::Point2f &np1 )
{
  Eigen::Vector3d x0(np0.x, np0.y, 1.0), x1(np1.x, np1.y, 1.0);
  Eigen::Vector3d l1 = ess_mat*x0;
  double essential_dist = x1.transpose()*l1;
  essential_dist *= essential_dist;
  // To have the (squared) euclidean distance, the epipolar line l1 should be in the form l1 = (a,b,-d)',
  // where a,b the unit 2D normal vector and d the line distance. Hence, we have to normalize l1 = ess_mat*x0
  double denom = l1(0)*l1(0) + l1(1)*l1(1);
  // If denom ~ 0, x0 is an epipole
  if( denom > std::numeric_limits< double >::epsilon() )
    return essential_dist/denom;
  else
    return 0.0;
}

CameraCalibrationBase::CameraCalibrationBase ( int cache_max_size, const std::string &cache_folder ) :
  cache_max_size_ ( cache_max_size ),
  cache_folder_ ( cache_folder )
{

  if ( cache_max_size_ < 1 )
    cache_max_size_ = 1;

  cache_folder_ += "/";
  cache_folder_ += cache_folder_basename;
  uuid_t uuid;
  char uuid_str[37];
  uuid_generate ( uuid );
  uuid_unparse_lower ( uuid, uuid_str );
  cache_folder_ += uuid_str;

  path cf_path ( cache_folder_ );
  if ( !create_directory ( cf_path ) )
  {
    throw std::runtime_error ( "CameraCalibrationBase::CameraCalibrationBase() : Failed to create the cache folder" );
  }
}

CameraCalibrationBase::~CameraCalibrationBase()
{
  clearDiskCache();
  path cf_path ( cache_folder_ );
  remove ( cf_path );
}

void CameraCalibrationBase::clearDiskCache()
{
  path cf_path ( cache_folder_ );
  if ( exists ( cf_path ) && is_directory ( cf_path ) )
  {
    // cf_path actually exist: delete all the previously cached files inside
    for ( directory_iterator d_iter ( cf_path ); d_iter!=directory_iterator(); d_iter++ )
    {
      if ( boost::starts_with ( d_iter->path().filename().string(), cache_file_basename ) )
      {
        remove ( d_iter->path() );
      }
    }
  }
}

void CameraCalibrationBase::cachePutImage ( int i, const cv::Mat &img )
{
  images_cache_.push_front ( pair<int,Mat> ( i,img.clone() ) );
  if ( int ( images_cache_.size() ) > cache_max_size_ )
  {
    // Save to disk the oldest image in the cache
    imwrite ( getCacheFilename ( images_cache_.back().first ), images_cache_.back().second );
    // .. and remove it from memory
    images_cache_.pop_back();
  }
}

Mat CameraCalibrationBase::cacheGetImage ( int i )
{
  // Look for the image
  auto it = images_cache_.begin();
  for ( ; it != images_cache_.end(); it++ )
    if ( it->first == i )
      break;

  Mat img;

  if ( it == images_cache_.begin() )
  {
    // Cache contains the i-th image and it is the newest element: already in the right position!
    img = it->second;
  }
  else if ( it != images_cache_.end() )
  {
    // Cache contains the i-th image: move it to front
    img = it->second;
    images_cache_.splice ( images_cache_.begin(), images_cache_, it, std::next ( it ) );
  }
  else
  {
    // Cache does not contain the i-th image: load it from file and push_to front

    img = imread ( getCacheFilename ( i ), cv::IMREAD_UNCHANGED );

    images_cache_.push_front ( pair<int,Mat> ( i,img ) );
    // In case, remove from memory the oldest image
    if ( int ( images_cache_.size() ) > cache_max_size_ )
    {
      string cache_fn = getCacheFilename ( images_cache_.back().first );
      path cf_path ( cache_fn );

      if ( !exists ( cf_path ) )
        // Not yet cached into disk
      {
        imwrite ( cache_fn, images_cache_.back().second );
      }

      images_cache_.pop_back();
    }
  }
  
  return img;
}

string CameraCalibrationBase::getCacheFilename ( int i )
{
  stringstream sstr;
  sstr<<cache_folder_;
  sstr<<"/";
  sstr<<cache_file_basename;
  sstr<<setfill ( '0' ) <<setw ( 8 ) <<i;
  sstr<<".png";

  return sstr.str();
}
void CameraCalibrationBase::setBoardSize( const cv::Size &s )
{
  cv_ext_assert(s.width > 0 && s.height > 0 && s.width != s.height );
  board_size_ = s;
}

CameraCalibration::CameraCalibration ( int cache_max_size, const std::string &cache_folder ) :
  CameraCalibrationBase ( cache_max_size, cache_folder )
{}

bool CameraCalibration::addImage ( const cv::Mat &img )
{
  cv_ext_assert(board_size_.width > 0 && board_size_.height > 0 && board_size_.width != board_size_.height );

  // Check image size and type
  if ( num_images_ )
  {
    if ( img.size() != image_size_ || img.type() != image_type_  )
      return false;
  }
  else if ( image_size_ != cv::Size ( -1,-1 ) )
    if ( img.size() != image_size_ ) 
      return false;
    
  Mat img_gray;
  if ( img.channels() == 3 )
  {
    cvtColor ( img, img_gray, cv::COLOR_BGR2GRAY );
  }
  else if ( img.channels() == 1 )
  {
    img_gray = img;
  }
  else
    // Unknown format
  {
    return false;
  }

  // Extract corners, otherwise return false
  vector<Point2f> img_pts;
  if ( !findCheckerboardCornersPyr ( img_gray, board_size_, img_pts, pyr_num_levels_, pattern_has_white_circle_ ) )
  {
    return false;
  }

  addImage(img, img_pts);

  return true;
}

void CameraCalibration::addImage( const cv::Mat &img, const std::vector<cv::Point2f> &corners_pts )
{
  cv_ext_assert(board_size_.width > 0 && board_size_.height > 0 && board_size_.width != board_size_.height );
  cv_ext_assert(static_cast<int>(corners_pts.size()) == board_size_.width*board_size_.height);

  // Store extracted corners
  images_points_.push_back (corners_pts );

  images_masks_.push_back ( true );
  per_view_errors_.push_back ( std::numeric_limits<double>::infinity() );

  // First image? Set image size and type
  if ( !num_images_ )
  {
    image_size_ = img.size();
    image_type_ = img.type();
  }

  // Add the image to cache
  cachePutImage ( num_images_, img );

  num_images_++;
  num_active_images_++;
}

bool CameraCalibration::addImageFile ( const string &filename )
{
  Mat img = imread ( filename, cv::IMREAD_UNCHANGED );
  if ( img.empty() )
  {
    return false;
  }

  return addImage ( img );
}

void CameraCalibration::setImageActive ( int i, bool active )
{
  if ( images_masks_[i] != active )
  {
    images_masks_[i] = active;
    num_active_images_ += active?1:-1;
  }
}

double CameraCalibration::calibrate()
{
  // Use only the selected images
  std::vector< std::vector<cv::Point2f> > calib_images_points;
  for ( int i = 0; i < num_images_; i++ )
  {
    calib_images_points.reserve(num_active_images_);
    if ( images_masks_[i] )
      calib_images_points.push_back ( images_points_[i] );
  }

  if ( !calib_images_points.size() )
  {
    return std::numeric_limits<double>::infinity();
  }

  vector<vector<Point3f> > object_points ( 1 );
  object_points[0] = getBoardCornerPositions(board_size_, square_size_);
  object_points.resize ( calib_images_points.size(), object_points[0] );

  int flags = 0;
  if ( fix_principal_point_ )
    flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
  if ( zero_tan_dist_ )
    flags |= cv::CALIB_ZERO_TANGENT_DIST;
  if ( fix_aspect_ratio_ )
    flags |= cv::CALIB_FIX_ASPECT_RATIO;
  if ( use_intrinsic_guess_ && !camera_matrix_.empty() )
    flags |= cv::CALIB_USE_INTRINSIC_GUESS;

  cv::TermCriteria term_criteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                  max_iter_, std::numeric_limits<double>::epsilon());
  vector<Mat> r_vecs, t_vecs;
  cv::calibrateCamera ( object_points, calib_images_points, image_size_, camera_matrix_,
                        dist_coeffs_, r_vecs, t_vecs,
                        flags | cv::CALIB_FIX_K4 |cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6,
                        term_criteria);

  // Compute reprojection error
  vector<Point2f> rep_image_points;
  int num_points = 0, i_calib = 0;
  double total_err = 0, err;
  int obj_pts_size = ( int ) object_points[0].size();

  for ( int i = 0; i < num_images_; i++ )
  {
    if ( images_masks_[i] )
    {
      projectPoints ( Mat ( object_points[0] ), r_vecs[i_calib], t_vecs[i_calib], camera_matrix_,
                      dist_coeffs_, rep_image_points );
      err = cv::norm ( Mat ( images_points_[i] ), Mat ( rep_image_points ), cv::NORM_L2SQR );

      per_view_errors_[i] = std::sqrt ( err/obj_pts_size );

      total_err += err;
      num_points += obj_pts_size;
      i_calib++;
    }
    else
    {
      per_view_errors_[i] = std::numeric_limits<double>::infinity();
    }
  }

  return std::sqrt ( total_err/num_points );
}

PinholeCameraModel CameraCalibration::getCamModel()
{
  if ( camera_matrix_.empty() )
    return PinholeCameraModel();
  else
    return PinholeCameraModel ( camera_matrix_, image_size_.width,
                                image_size_.height, dist_coeffs_ );
}

void CameraCalibration::setCamModel ( const cv_ext::PinholeCameraModel &model )
{
  camera_matrix_ = model.cameraMatrix();
  dist_coeffs_ = model.distorsionCoeff();
  image_size_ = model.imgSize();
}

void CameraCalibration::computeAverageExtrinsicParameters ( cv::Mat &r_vec, cv::Mat &t_vec )
{
  vector<Point3f> object_points, corner_pos;
  vector<Point2f> image_points;

  corner_pos = getBoardCornerPositions(board_size_, square_size_);
  object_points.reserve ( corner_pos.size() *num_images_ );
  image_points.reserve ( corner_pos.size() *num_images_ );
  for ( int i = 0; i < num_images_; i++ )
  {
    if ( images_masks_[i] )
    {
      object_points.insert ( object_points.end(), corner_pos.begin(), corner_pos.end() );
      image_points.insert ( image_points.end(), images_points_[i].begin(), images_points_[i].end() );
    }
  }

  solvePnP ( object_points, image_points, camera_matrix_, dist_coeffs_, r_vec, t_vec );
}

std::vector<cv::Point2f>  CameraCalibration::getCorners( int i )
{
  return images_points_[i];
}

void CameraCalibration::getCornersImage ( int i, cv::Mat &corners_img, float scale_factor )
{
  Mat img = cacheGetImage ( i );
  
  if ( scale_factor > 1 )
  {
    float scale = 1.0f/scale_factor;
    std::vector<cv::Point2f> img_pts;
    for ( auto &p : images_points_[i] )
      img_pts.push_back ( scale*p );
    
    if ( img.channels() < 3 )
      cvtColor ( img, img, COLOR_GRAY2BGR );
    
    cv::resize ( img, corners_img, cv::Size(), scale, scale );
    drawChessboardCorners ( corners_img, board_size_, Mat ( img_pts ), true );
  }
  else
  {
    if ( img.channels() < 3 )
      cvtColor ( img, corners_img, COLOR_GRAY2BGR );
    else
      img.copyTo(corners_img);

    drawChessboardCorners ( corners_img, board_size_, Mat ( images_points_[i] ), true );
  }
}

void CameraCalibration::getUndistortedImage ( int i, cv::Mat &und_img, float scale_factor )
{
  if ( camera_matrix_.empty() || dist_coeffs_.empty() )
    return;

  Mat img = cacheGetImage ( i );

  if ( scale_factor > 1 )
  {
    Mat tmp_und_img;
    cv::undistort ( img, tmp_und_img, camera_matrix_, dist_coeffs_ );
    float scale = 1.0f/scale_factor;
    cv::resize ( tmp_und_img, und_img, cv::Size(), scale, scale );
  }
  else
    cv::undistort ( img, und_img, camera_matrix_, dist_coeffs_ );
}

void CameraCalibration::getCornersDistribution ( float kernel_stdev, cv::Mat &corner_dist, float scale_factor )
{
  if ( scale_factor < 1 )
    scale_factor = 1.0f;

  float scale = 1.0f/scale_factor;
  Mat accumulator = Mat::zeros ( Size ( scale*image_size_.width, scale*image_size_.height ),
                                 cv::DataType<float>::type );

  if( corner_dist.type() != accumulator.type() || corner_dist.size() != accumulator.size() )
    corner_dist.create(accumulator.size(), accumulator.type() );

  if( scale_factor > 1 )
  {
    for ( int i = 0; i < int ( images_points_.size() ); i++ )
    {
      if ( images_masks_[i] )
      {
        for ( auto &p : images_points_[i] )
          accumulator.at<float> ( scale*p.y, scale*p.x ) += 1.0f;
      }
    }
  }
  else
  {
    for ( int i = 0; i < int ( images_points_.size() ); i++ )
    {
      if ( images_masks_[i] )
      {
        for ( auto &p : images_points_[i] )
          accumulator.at<float> ( p.y, p.x ) += 1.0f;
      }
    }    
  }

  kernel_stdev *= scale;
  int kernel_size = cvRound ( kernel_stdev*4*2 + 1 ) |1;
  Mat kernel1D = cv::getGaussianKernel ( kernel_size, kernel_stdev, CV_32F );
  cv::normalize ( kernel1D,kernel1D, 0.0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type );
  cv::sepFilter2D ( accumulator, corner_dist, -1, kernel1D, kernel1D );
}

void CameraCalibration::clear() 
{
  clearDiskCache();

  num_images_ = num_active_images_ = 0;  
  
  camera_matrix_ = cv::Mat();
  dist_coeffs_ = cv::Mat();
  
  images_points_.clear();
  images_masks_.clear();
  per_view_errors_.clear();
}

StereoCameraCalibration::StereoCameraCalibration ( int cache_max_size, const std::string &cache_folder ) :
  CameraCalibrationBase ( cache_max_size, cache_folder )
{}

void StereoCameraCalibration::setCamModels ( const std::vector < PinholeCameraModel > &cam_models )
{
  cv_ext_assert( cam_models.size() == 2 );
  cv_ext_assert( cam_models[0].imgSize() == cam_models[1].imgSize() );

  image_size_ = cam_models[0].imgSize();
  for ( int k = 0; k < 2; k++ )
  {
    camera_matrices_[k] = cam_models[k].cameraMatrix();
    dist_coeffs_[k] = cam_models[k].distorsionCoeff();
  }
}

void StereoCameraCalibration::getExtrinsicsParameters ( Mat& r_mat, Mat& t_vec )
{ 
  r_mat = r_mat_.clone();
  t_vec = t_vec_.clone();
}

std::vector < PinholeCameraModel > StereoCameraCalibration::getCamModels()
{
  std::vector < PinholeCameraModel > cam_models;
  cam_models.reserve(2);
  if ( !camera_matrices_[0].empty() && !camera_matrices_[1].empty() )
  {
    for ( int k = 0; k < 2; k++ )
      cam_models.emplace_back( camera_matrices_[k], image_size_.width,
                               image_size_.height, dist_coeffs_[k] );
  }
  else
  {
    for ( int k = 0; k < 2; k++ )
      cam_models.emplace_back();
  }
  return cam_models;
}

bool StereoCameraCalibration::addImagePair ( const std::vector< cv::Mat > &imgs )
{
  cv_ext_assert(board_size_.width > 0 && board_size_.height > 0 && board_size_.width != board_size_.height );
  cv_ext_assert(imgs.size() == 2);

  // Check image size and type
  if ( num_pairs_ )
  {
    if ( imgs[0].size() != image_size_ || imgs[1].size() != image_size_ || 
         imgs[0].type() != image_type_ || imgs[1].type() != image_type_ )
      return false;
  }
  else if ( image_size_ != cv::Size ( -1,-1 ) )
  {
    if ( imgs[0].size() != image_size_ || imgs[1].size() != image_size_ )
      return false;
  }

  vector< vector<Point2f> > img_pts(2);
  for ( int k = 0; k < 2; k++ )
  {
    Mat img_gray;
    if ( imgs[k].channels() == 3 )
    {
      cvtColor ( imgs[k], img_gray, cv::COLOR_BGR2GRAY );
    }
    else if ( imgs[k].channels() == 1 )
    {
      img_gray = imgs[k];
    }
    else
      // Unknown format
    {
      return false;
    }

    // Extract corners, otherwise return false
    if ( !findCheckerboardCornersPyr ( img_gray, board_size_, img_pts[k], pyr_num_levels_, pattern_has_white_circle_ ) )
    {
      return false;
    }
  }

  addImagePair(imgs, img_pts);

  return true;
}

void StereoCameraCalibration::addImagePair( const std::vector< cv::Mat > &imgs,
                                            const std::vector< std::vector<cv::Point2f> > &corners_pts )
{
  cv_ext_assert(board_size_.width > 0 && board_size_.height > 0 && board_size_.width != board_size_.height );
  cv_ext_assert(corners_pts.size() == 2);
  cv_ext_assert(static_cast<int>(corners_pts[0].size()) == board_size_.width*board_size_.height);
  cv_ext_assert(static_cast<int>(corners_pts[1].size()) == board_size_.width*board_size_.height);

  // Store extracted corners
  for ( int k = 0; k < 2; k++ )
  {
    images_points_[k].push_back ( corners_pts[k] );
  }

  pairs_masks_.push_back ( true );
  per_view_errors_.push_back ( std::numeric_limits<double>::infinity() );

  // First images? Set image size and type
  if ( !num_pairs_ )
  {
    image_size_ = imgs[0].size();
    image_type_ = imgs[0].type();
  }

  // Add the images to cache
  for ( int k = 0; k < 2; k++ )
    cachePutImage ( 2*num_pairs_ + k, imgs[k] );

  num_pairs_++;
  num_active_pairs_++;
}

bool StereoCameraCalibration::addImagePairFiles ( const std::vector< std::string > &filenames )
{
  cv_ext_assert(filenames.size() == 2);

  std::vector< Mat > imgs(2);
  for ( int k = 0; k < 2; k++ )
  {
    imgs[k] = imread ( filenames[k], cv::IMREAD_UNCHANGED );
    if ( imgs[k].empty() )
      return false;
  }

  return addImagePair ( imgs );
}

void StereoCameraCalibration::setImagePairActive ( int i, bool active )
{
  if ( pairs_masks_[i] != active )
  {
    pairs_masks_[i] = active;
    num_active_pairs_ += active?1:-1;
  }
}

double StereoCameraCalibration::calibrate()
{
  // Use only the selected pairs
  std::vector< std::vector<cv::Point2f> > calib_images_points[2];
  for ( int k = 0; k < 2; k++ )
    calib_images_points[k].reserve(num_active_pairs_);
  
  for ( int i = 0; i < num_pairs_; i++ )
  {
    if ( pairs_masks_[i] )
    {
      for ( int k = 0; k < 2; k++ )
        calib_images_points[k].push_back ( images_points_[k][i] );
    }
  }

  if ( !calib_images_points[0].size() )
    return std::numeric_limits<double>::infinity();

  vector<vector<Point3f> > object_points ( 1 );
  object_points[0] = getBoardCornerPositions(board_size_, square_size_);
  object_points.resize ( calib_images_points[0].size(), object_points[0] );

  int flags = 0;
  if ( camera_matrices_[0].empty() || camera_matrices_[1].empty() ||
       use_intrinsic_guess_ )
  {
    if ( fix_principal_point_ )
      flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    if ( zero_tan_dist_ )
      flags |= cv::CALIB_ZERO_TANGENT_DIST;
    if ( fix_aspect_ratio_ )
      flags |= cv::CALIB_FIX_ASPECT_RATIO;
    if ( force_same_focal_lenght_ )
      flags |= cv::CALIB_SAME_FOCAL_LENGTH;
    if ( use_intrinsic_guess_ && !camera_matrices_[0].empty() && !camera_matrices_[1].empty() )
      flags |= cv::CALIB_USE_INTRINSIC_GUESS;

    flags |= cv::CALIB_FIX_K4|cv::CALIB_FIX_K5|cv::CALIB_FIX_K6;
  }
  else
  {
    flags |= cv::CALIB_FIX_INTRINSIC;
  }

  cv::TermCriteria term_criteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                  max_iter_, std::numeric_limits<double>::epsilon());

  Mat ess_mat, fund_mat;
  cv::stereoCalibrate ( object_points, calib_images_points[0], calib_images_points[1],
                        camera_matrices_[0], dist_coeffs_[0], camera_matrices_[1], dist_coeffs_[1],
                        image_size_, r_mat_, t_vec_, ess_mat, fund_mat, flags, term_criteria );

  auto cam_models = getCamModels();
  stereo_rect_.setCameraParameters(cam_models, r_mat_, t_vec_ );
  stereo_rect_.update();

  /* Compute the RMS error using the distances between extracted points and epipolar line,
   * using the epipolar geometry constraint: m2^t*F*m1=0 */

  Eigen::Matrix3d eig_fund_mat, eig_fund_mat_t;
  for(int r = 0; r < 3; r++)
    for(int c = 0; c < 3; c++)
      eig_fund_mat(r, c) = fund_mat.at<double>(r, c);

  eig_fund_mat_t = eig_fund_mat.transpose();

  int total_num_points = 0;
  double total_err = 0;
  for ( int i = 0; i < num_pairs_; i++ )
  {
    if (pairs_masks_[i])
    {
      int npt = (int) images_points_[0][i].size();
      std::vector<cv::Point2f> img_pts[2];
      for (int k = 0; k < 2; k++)
        cv::undistortPoints( images_points_[k][i], img_pts[k], camera_matrices_[k], dist_coeffs_[k],
                             Mat(), camera_matrices_[k]);

      double err = 0;
      // Errors in camera 1
      for ( int j = 0; j < npt; j++ )
        err += computeEpipolarSquareDistance (eig_fund_mat, img_pts[0][j], img_pts[1][j] );

      // Errors in camera 0
      for ( int j = 0; j < npt; j++ )
        err += computeEpipolarSquareDistance (eig_fund_mat_t, img_pts[1][j], img_pts[0][j] );

      total_err += err;
      total_num_points += 2*npt;
      per_view_errors_[i] = std::sqrt ( err/(2*npt) );
    }
    else
    {
      per_view_errors_[i] = std::numeric_limits<double>::infinity();
    }
  }

  return std::sqrt ( total_err/total_num_points );
}

std::vector< std::vector< cv::Point2f > > StereoCameraCalibration::getCornersPair( int i )
{
  std::vector< std::vector< cv::Point2f > > corners(2);
  corners[0] = images_points_[0][i];
  corners[1] = images_points_[1][i];

  return corners;
}

void StereoCameraCalibration::getCornersImagePair ( int i, std::vector< cv::Mat > &corners_imgs, float scale_factor )
{
  corners_imgs.resize(2);
  for( int k = 0; k < 2; k++ )
  {
    Mat img = cacheGetImage ( 2*i + k );
    
    if ( scale_factor > 1 )
    {
      float scale = 1.0f/scale_factor;
      std::vector<cv::Point2f> img_pts;
      for ( auto &p : images_points_[k][i] )
        img_pts.push_back ( scale*p );
      
      if ( img.channels() < 3 )
        cvtColor ( img, img, COLOR_GRAY2BGR );
      
      cv::resize ( img, corners_imgs[k], cv::Size(), scale, scale );
      drawChessboardCorners ( corners_imgs[k], board_size_, Mat ( img_pts ), true );
    }
    else
    {
      if ( img.channels() < 3 )
        cvtColor ( img, corners_imgs[k], COLOR_GRAY2BGR );
      else
        img.copyTo(corners_imgs[k]);

      drawChessboardCorners ( corners_imgs[k], board_size_, Mat ( images_points_[k][i] ), true );
    }
  }
}

void StereoCameraCalibration::getCornersDistribution ( float kernel_stdev, std::vector< cv::Mat > &corner_dists,
                                                       float scale_factor )
{
  corner_dists.resize(2);

  if ( scale_factor < 1 )
    scale_factor = 1.0f;

  float scale = 1.0f/scale_factor;
  
  kernel_stdev *= scale;
  int kernel_size = cvRound ( kernel_stdev*4*2 + 1 ) |1;
  Mat kernel1D = cv::getGaussianKernel ( kernel_size, kernel_stdev, CV_32F );
  cv::normalize ( kernel1D,kernel1D, 0.0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type );
  
  
  for( int k = 0; k < 2; k++ )
  {
    Mat accumulator = Mat::zeros ( Size ( scale*image_size_.width, scale*image_size_.height ),
                                  cv::DataType<float>::type );

    if( scale_factor > 1 )
    {
      for ( int i = 0; i < int ( images_points_[k].size() ); i++ )
      {
        if ( pairs_masks_[i] )
        {
          for ( auto &p : images_points_[k][i] )
            accumulator.at<float> ( scale*p.y, scale*p.x ) += 1.0f;
        }
      }
    }
    else
    {
      for ( int i = 0; i < int ( images_points_[k].size() ); i++ )
      {
        if ( pairs_masks_[i] )
        {
          for ( auto &p : images_points_[k][i] )
            accumulator.at<float> ( p.y, p.x ) += 1.0f;
        }
      }
    }
    cv::sepFilter2D ( accumulator, corner_dists[k], -1, kernel1D, kernel1D );
  }
}

void StereoCameraCalibration::rectifyImagePair ( int i, std::vector< cv::Mat > &rect_imgs, float scale_factor )
{
  std::vector< cv::Mat > imgs(2);
  rect_imgs.resize(2);

  for( int k = 0; k < 2; k++ )
    imgs[k] = cacheGetImage ( 2*i + k );
  
  if ( scale_factor > 1 )
  {
    float scale = 1.0f/scale_factor;
    std::vector < Mat > tmp_rect_imgs;
    stereo_rect_.rectifyImagePair( imgs, tmp_rect_imgs );

    for( int k = 0; k < 2; k++ )
      cv::resize ( tmp_rect_imgs[k], rect_imgs[k], cv::Size(), scale, scale );
  }
  else
  {
    stereo_rect_.rectifyImagePair( imgs, rect_imgs );
  }
}

void StereoCameraCalibration::clear()
{
  clearDiskCache();

  num_pairs_ = num_active_pairs_ = 0;
  force_same_focal_lenght_ = false;
  
  for( int k = 0; k < 2; k++ )
  {
    images_points_[k].clear();
    camera_matrices_[k] = cv::Mat();
    dist_coeffs_[k] = cv::Mat();
  }
  
  r_mat_ = cv::Mat();
  t_vec_ = cv::Mat();

  pairs_masks_.clear();
  per_view_errors_.clear();

  stereo_rect_ = cv_ext::StereoRectification();
}

MultiStereoCameraCalibration::MultiStereoCameraCalibration( int num_cameras, int cache_max_size,
                                                            const std::string cache_folder) :
    CameraCalibrationBase ( cache_max_size, cache_folder ),
    num_cameras_(num_cameras)
{
  cv_ext_assert( num_cameras_ > 1 );
  cam_models_.resize(num_cameras);
  r_mats_.resize(num_cameras);
  t_vecs_.resize(num_cameras);
  images_points_.resize(num_cameras);
}

void MultiStereoCameraCalibration::setCamModels(const std::vector<PinholeCameraModel> &cam_models)
{
  cv_ext_assert( static_cast<int>( cam_models.size() ) == num_cameras_ );
  for(int k = 1; k < num_cameras_; k++ )
    cv_ext_assert( cam_models[0].imgSize() == cam_models[k].imgSize() );

  image_size_ = cam_models[0].imgSize();
  cam_models_ = cam_models;
}

void MultiStereoCameraCalibration::getExtrinsicsParameters(std::vector<cv::Mat> &r_mats, std::vector<cv::Mat> &t_vecs)
{
  r_mats.resize(num_cameras_);
  t_vecs.resize(num_cameras_);

  for(int k = 0; k < num_cameras_; k++ )
  {
    r_mats[k] = r_mats_[k].clone();
    t_vecs[k] = t_vecs_[k].clone();
  }
}

bool MultiStereoCameraCalibration::addImageTuple(const std::vector<cv::Mat> &imgs )
{
  cv_ext_assert( board_size_.width > 0 && board_size_.height > 0 && board_size_.width != board_size_.height );
  cv_ext_assert( static_cast<int>( imgs.size() ) == num_cameras_ );

  // Check image size and type
  if ( num_tuples_ )
  {
    for( int k = 0; k < num_cameras_; k++ )
    {
      if ( imgs[k].size() != image_size_ || imgs[k].type() != image_type_ )
        return false;
    }
  }
  else if ( image_size_ != cv::Size ( -1,-1 ) )
  {
    for (int k = 0; k < num_cameras_; k++)
    {
      if (imgs[k].size() != image_size_)
        return false;
    }
  }

  vector< vector<Point2f> > img_pts(num_cameras_);
  int good_checkerboard = 0;
  for ( int k = 0; k < num_cameras_; k++ )
  {
    Mat img_gray;
    if ( imgs[k].channels() == 3 )
    {
      cvtColor ( imgs[k], img_gray, cv::COLOR_BGR2GRAY );
    }
    else if ( imgs[k].channels() == 1 )
    {
      img_gray = imgs[k];
    }
    else
    // Unknown format
    {
      return false;
    }

    // Extract corners, otherwise return false
    if ( findCheckerboardCornersPyr ( img_gray, board_size_, img_pts[k], pyr_num_levels_, pattern_has_white_circle_ ) )
      good_checkerboard++;
  }

  if( good_checkerboard < 2 )
    return false;

  addImageTuple(imgs, img_pts);

  return true;
}

void MultiStereoCameraCalibration::addImageTuple(const std::vector<cv::Mat> &imgs,
                                                 const std::vector<std::vector<cv::Point2f> > &corners_pts)
{
  cv_ext_assert( board_size_.width > 0 && board_size_.height > 0 && board_size_.width != board_size_.height );
  cv_ext_assert( static_cast<int>( imgs.size() ) == num_cameras_ );
  for(int k = 0; k < num_cameras_; k++ )
    cv_ext_assert( static_cast<int>(corners_pts[k].size()) == board_size_.width*board_size_.height ||
                   static_cast<int>(corners_pts[k].size()) == 0 );

  // Store extracted corners
  for(int k = 0; k < num_cameras_; k++ )
    images_points_[k].push_back ( corners_pts[k] );

  tuples_masks_.push_back ( true );
  per_view_errors_.push_back ( std::numeric_limits<double>::infinity() );

  // First images? Set image size and type
  if ( !num_tuples_ )
  {
    image_size_ = imgs[0].size();
    image_type_ = imgs[0].type();
  }

  // Add the images to cache
  for ( int k = 0; k < num_cameras_; k++ )
    cachePutImage ( num_cameras_*num_tuples_ + k, imgs[k] );

  num_tuples_++;
  num_active_tuples_++;
}

bool MultiStereoCameraCalibration::addImageTupleFiles( const std::vector< std::string > &filenames )
{
  cv_ext_assert( static_cast<int>( filenames.size() ) == num_cameras_);

  std::vector< Mat > imgs(num_cameras_);
  for ( int k = 0; k < num_cameras_; k++ )
  {
    imgs[k] = imread ( filenames[k], cv::IMREAD_UNCHANGED );
    if ( imgs[k].empty() )
      return false;
  }

  return addImageTuple(imgs);
}

void MultiStereoCameraCalibration::setImageTupleActive(int i, bool active)
{
  if ( tuples_masks_[i] != active )
  {
    tuples_masks_[i] = active;
    num_active_tuples_ += active?1:-1;
  }
}

double MultiStereoCameraCalibration::calibrate()
{
  if ( !num_active_tuples_ )
    return std::numeric_limits<double>::infinity();

  std::vector< std::vector< cv::Mat > > rel_ptn_r_vecs(num_active_tuples_, std::vector< cv::Mat >(num_cameras_)),
                                        rel_ptn_t_vecs(num_active_tuples_, std::vector< cv::Mat >(num_cameras_));

  std::vector<cv::Point3f> corners = getBoardCornerPositions(board_size_, square_size_);

  for(int i = 0, t = 0; i < num_tuples_; i++ )
  {
    if( tuples_masks_[i] )
    {
      for(int c = 0; c < num_cameras_; c++ )
      {
        if( images_points_[c][i].size() )
        {
          cv::solvePnP(corners, images_points_[c][i], cam_models_[c].cameraMatrix(), cam_models_[c].distorsionCoeff(),
                       rel_ptn_r_vecs[t][c], rel_ptn_t_vecs[t][c] );
        }
      }
      t++;
    }
  }

  // Rigid body transformations between couples of cameras.
  // The [i][j] transformation transforms 3D points referred to the j-th camera reference frame into
  // points referred to the i-th camera reference frame
  std::vector< std::vector< cv::Mat > > rel_cam_r_vecs(num_cameras_, std::vector< cv::Mat >(num_cameras_)),
                                        rel_cam_t_vecs(num_cameras_, std::vector< cv::Mat >(num_cameras_));

  // Related adjacency mattrix
  cv::Mat_<double > adj_mat(num_cameras_,num_cameras_);
  adj_mat.setTo(0);

  // Vector used to extract the medians
  std::vector< std::vector< double > > rt_vals;
  for( int i = 0; i < num_cameras_; i++ )
  {
    for( int j = i + 1; j < num_cameras_; j++ )
    {
      cv::Mat_<double> r_mat_i(3,3), r_mat_j(3,3), r_mat_ij(3,3), r_vec_ij(3,1), t_vec_ij(3,1);
      rt_vals.clear();
      rt_vals.resize(6);
      for( int t = 0; t < num_active_tuples_; t++ )
      {
        // g_{i,j} : [ R_i * R'_j, -R_i * R'_j * t_j + t_i ]
        if( !rel_ptn_r_vecs[t][i].empty() && !rel_ptn_r_vecs[t][j].empty() )
        {
          cv_ext::angleAxis2RotMat<double>( rel_ptn_r_vecs[t][i], r_mat_i ) ;
          cv_ext::angleAxis2RotMat<double>( rel_ptn_r_vecs[t][j], r_mat_j );

          r_mat_ij = r_mat_i*r_mat_j.t();
          t_vec_ij = -r_mat_ij*rel_ptn_t_vecs[t][j] + rel_ptn_t_vecs[t][i];

          cv_ext::rotMat2AngleAxis<double>( r_mat_ij, r_vec_ij );

          for( int h = 0; h < 3; h++ )
          {
            rt_vals[h].push_back(r_vec_ij(h));
            rt_vals[3+h].push_back(t_vec_ij(h));
          }
        }
      }

      if( rt_vals[0].size() )
      {
        for( int h = 0; h < 3; h++ )
        {
          r_vec_ij(h)= getMedian( rt_vals[h] );
          t_vec_ij(h)= getMedian( rt_vals[3+h] );
        }

        rel_cam_r_vecs[i][j] = r_vec_ij.clone();
        rel_cam_t_vecs[i][j] = t_vec_ij.clone();
        cv_ext::angleAxis2RotMat<double>( r_vec_ij, r_mat_ij );
        r_mat_ij = r_mat_ij.t();
        cv_ext::rotMat2AngleAxis<double>( r_mat_ij, r_vec_ij );
        rel_cam_r_vecs[j][i] = r_vec_ij.clone();
        rel_cam_t_vecs[j][i] = -r_mat_ij*t_vec_ij;

        // An high value in the adjacency matrix means many covisible checkerboards -> more robust measurements
        adj_mat(i,j) = adj_mat(j,i) = 1.0/rt_vals[0].size();
      }
    }
  }

  auto paths = bellmanFordShortestPath(adj_mat, 0);


  // Use the shortest paths to compose the camera transformations
  for( int i = 0; i < num_cameras_; i++ )
  {
    r_mats_[i] = cv::Mat_<double>::eye(cv::Size(3,3));
    t_vecs_[i] = (cv::Mat_<double>(3, 1) << 0, 0, 0);

    if( static_cast<int>(paths[i].size()) >= 2 )
    {
      for( int j = paths[i].size() - 1; j > 0; j-- )
      {
        cv::Mat_<double> rel_r_mat(3, 3);
        cv_ext::angleAxis2RotMat<double>( rel_cam_r_vecs[paths[i][j-1]][paths[i][j]], rel_r_mat );
        r_mats_[i] = rel_r_mat*r_mats_[i];
        t_vecs_[i] = rel_r_mat*t_vecs_[i] + rel_cam_t_vecs[paths[i][j-1]][paths[i][j]];
      }
    }
  }

  std::vector<double> params(6*(num_cameras_+ num_active_tuples_ - 1));
  double *cam_params = params.data(), *ptn_params = params.data() + 6*(num_cameras_ - 1);

  // Copy the num_cameras_ - 1 camera transformations in the parameters vector
  for(int c = 0; c < num_cameras_ - 1; c++ )
  {
    cv::Mat_<double> r_vec(3, 1);
    cv_ext::rotMat2AngleAxis<double>(r_mats_[c+1], r_vec);
    double *cam_param = cam_params + 6*c;
    for( int j = 0; j < 3; j++ )
    {
      cam_param[j] = r_vec(j);
      cam_param[j+3] = t_vecs_[c+1].at<double>(j);
    }
  }

  // Extract and copy the num_active_tuples_ pattern transformations in the parameters vector
  for(int t = 0; t < num_active_tuples_; t++ )
  {
    rt_vals.clear();
    rt_vals.resize(6);
    cv::Mat_<double> r_mat_ptn(3,3), r_vec_ptn(3,1), t_vec_ptn(3,1);
    for(int c = 0; c < num_cameras_; c++ )
    {
      if( !rel_ptn_r_vecs[t][c].empty() )
      {
        cv_ext::angleAxis2RotMat<double>(rel_ptn_r_vecs[t][c], r_mat_ptn );

        r_mat_ptn = r_mats_[c].t()*r_mat_ptn;
        t_vec_ptn = r_mats_[c].t()*(rel_ptn_t_vecs[t][c] - t_vecs_[c]);

        cv_ext::rotMat2AngleAxis<double>( r_mat_ptn, r_vec_ptn );

        for( int h = 0; h < 3; h++ )
        {
          rt_vals[h].push_back(r_vec_ptn(h));
          rt_vals[3+h].push_back(t_vec_ptn(h));
        }
      }
    }

    double *ptn_param = ptn_params + 6*t;

    for( int h = 0; h < 6; h++ )
      ptn_param[h] = getMedian( rt_vals[h] );
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = max_iter_;
  options.gradient_tolerance = std::numeric_limits<double>::epsilon();
  options.function_tolerance = std::numeric_limits<double>::epsilon();

  ceres::Problem problem;

  std::vector< Eigen::Vector3d > pattern_pts;
  pattern_pts.reserve(corners.size());
  for( auto &cp : corners )
    pattern_pts.emplace_back( cp.x, cp.y, cp.z );

  double observed_pt_data[2];
  Eigen::Map<Eigen::Vector2d> observed_pt(observed_pt_data);

  for( int i = 0, t = 0; i < num_tuples_; i++ )
  {
    if( tuples_masks_[i] )
    {
      for(int c = 0; c < num_cameras_; c++ )
      {
        if( images_points_[c][i].size() )
        {
          for( int k = 0; k < static_cast<int>(pattern_pts.size()); k++ )
          {
            observed_pt_data[0] = images_points_[c][i][k].x;
            observed_pt_data[1] = images_points_[c][i][k].y;

            ceres::HuberLoss* loss_function( new ceres::HuberLoss( huber_loss_alpha_ ) );
            if( c )
            {
              ceres::CostFunction *cost_function =
                  CalibReprojectionError::Create( cam_models_[c], pattern_pts[k], observed_pt);
              problem.AddResidualBlock(cost_function, loss_function, cam_params + 6*(c - 1), ptn_params + 6*t );
            }
            else
            {
              ceres::CostFunction *cost_function =
                  CalibReferenceCalibReprojectionError::Create( cam_models_[c], pattern_pts[k], observed_pt);
              problem.AddResidualBlock(cost_function, loss_function, ptn_params + 6*t );
            }
          }
        }
      }
      t++;
    }
  }

  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
//  std::cout << summary.FullReport() << "\n";

  // Copy the optimized camera transformations
  for(int c = 0; c < num_cameras_ - 1; c++ )
  {
    double *cam_param = cam_params + 6*c;
    ceres::AngleAxisToRotationMatrix( cam_param, ceres::RowMajorAdapter3x3((double *)r_mats_[c + 1].data) );

    for( int j = 0; j < 3; j++ )
      t_vecs_[c + 1].at<double>(j) = cam_param[3 + j];
  }

  std::vector< std::vector< Eigen::Matrix3d > > fund_mats(num_cameras_, std::vector< Eigen::Matrix3d >(num_cameras_));

  // Compute the relative fundamental matrices from the optimized transformation
  for( int i = 0; i < num_cameras_; i++ )
  {
    for (int j = i + 1; j < num_cameras_; j++)
    {
      cv::Mat_<double> r_mat_ij(3, 3), t_vec_ij(3, 1), ess_mat(3, 3), fund_mat(3, 3);

      r_mat_ij = r_mats_[i]*r_mats_[j].t();
      t_vec_ij = -r_mat_ij*t_vecs_[j] + t_vecs_[i];

      double *t = reinterpret_cast<double *>(t_vec_ij.data);
      cv::Mat_<double> skew_t = (cv::Mat_<double>(3, 3) <<  0,     -t[2], t[1],
                                                            t[2],  0,     -t[0],
                                                            -t[1], t[0],  0  );

      ess_mat = skew_t*r_mat_ij;
      fund_mat = (cam_models_[i].cameraMatrix().inv().t()) * ess_mat * (cam_models_[j].cameraMatrix().inv());

      if( fabs( fund_mat(2,2) ) > 0 )
        fund_mat *= 1./fund_mat(2,2);

      for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
          fund_mats[i][j](r, c) = fund_mat(r, c);

      fund_mats[j][i] = fund_mats[i][j].transpose();
    }
  }

  int total_num_points = 0;
  double total_err = 0;
  for( int i = 0; i < num_tuples_; i++ )
  {
    if( tuples_masks_[i] )
    {
      std::vector<cv::Point2f> img_pts[2];
      double err = 0;
      int npt = 0;
      for(int ci = 0; ci < num_cameras_; ci++ )
      {
        if( images_points_[ci][i].size() )
        {
          cv::undistortPoints( images_points_[ci][i], img_pts[0], cam_models_[ci].cameraMatrix(),
                               cam_models_[ci].distorsionCoeff(), Mat(), cam_models_[ci].cameraMatrix());

          for (int cj = ci + 1; cj < num_cameras_; cj++)
          {
            if( images_points_[cj][i].size() )
            {
              cv::undistortPoints( images_points_[cj][i], img_pts[1], cam_models_[cj].cameraMatrix(),
                                   cam_models_[cj].distorsionCoeff(), Mat(), cam_models_[cj].cameraMatrix());

              // Errors in camera j
              for ( int j = 0; j < static_cast<int>(img_pts[0].size()); j++ )
                err += computeEpipolarSquareDistance (fund_mats[cj][ci], img_pts[0][j], img_pts[1][j] );

              // Errors in camera i
              for ( int j = 0; j < static_cast<int>(img_pts[0].size()); j++ )
                err += computeEpipolarSquareDistance (fund_mats[ci][cj], img_pts[1][j], img_pts[0][j] );

              npt += 2*static_cast<int>(img_pts[0].size());
            }
          }
        }
      }
      total_err += err;
      total_num_points += npt;
      per_view_errors_[i] = std::sqrt ( err/npt );
    }
    else
    {
      per_view_errors_[i] = std::numeric_limits<double>::infinity();
    }
  }

  return std::sqrt ( total_err/total_num_points );
}

void MultiStereoCameraCalibration::getCornersImageTuple( int i, std::vector< cv::Mat > &corners_imgs,
                                                         float scale_factor)
{
  corners_imgs.resize(num_cameras_);
  for( int k = 0; k < num_cameras_; k++ )
  {
    Mat img = cacheGetImage ( num_cameras_*i + k );

    if ( scale_factor > 1 )
    {
      float scale = 1.0f/scale_factor;
      if ( img.channels() < 3 )
        cvtColor ( img, img, COLOR_GRAY2BGR );
      cv::resize ( img, corners_imgs[k], cv::Size(), scale, scale );

      if(images_points_[k][i].size())
      {
        std::vector<cv::Point2f> img_pts;
        for ( auto &p : images_points_[k][i] )
          img_pts.push_back ( scale*p );

        drawChessboardCorners ( corners_imgs[k], board_size_, Mat ( img_pts ), true );
      }
    }
    else
    {
      if ( img.channels() < 3 )
        cvtColor ( img, corners_imgs[k], COLOR_GRAY2BGR );
      else
        img.copyTo(corners_imgs[k]);
      if(images_points_[k][i].size())
        drawChessboardCorners ( corners_imgs[k], board_size_, Mat ( images_points_[k][i] ), true );
    }
  }
}

void MultiStereoCameraCalibration::getCornersDistribution( float kernel_stdev, std::vector<cv::Mat> &corner_dists,
                                                           float scale_factor)
{
  corner_dists.resize(num_cameras_);

  if ( scale_factor < 1 )
    scale_factor = 1.0f;

  float scale = 1.0f/scale_factor;

  kernel_stdev *= scale;
  int kernel_size = cvRound ( kernel_stdev*4*2 + 1 ) |1;
  Mat kernel1D = cv::getGaussianKernel ( kernel_size, kernel_stdev, CV_32F );
  cv::normalize ( kernel1D,kernel1D, 0.0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type );

  for( int k = 0; k < num_cameras_; k++ )
  {
    Mat accumulator = Mat::zeros ( Size ( scale*image_size_.width, scale*image_size_.height ),
                                   cv::DataType<float>::type );

    if( scale_factor > 1 )
    {
      for ( int i = 0; i < int ( images_points_[k].size() ); i++ )
      {
        if ( tuples_masks_[i] )
        {
          for ( auto &p : images_points_[k][i] )
            accumulator.at<float> ( scale*p.y, scale*p.x ) += 1.0f;
        }
      }
    }
    else
    {
      for ( int i = 0; i < int ( images_points_[k].size() ); i++ )
      {
        if ( tuples_masks_[i] )
        {
          for ( auto &p : images_points_[k][i] )
            accumulator.at<float> ( p.y, p.x ) += 1.0f;
        }
      }
    }
    cv::sepFilter2D ( accumulator, corner_dists[k], -1, kernel1D, kernel1D );
  }
}

void MultiStereoCameraCalibration::clear()
{
  clearDiskCache();

  num_tuples_ = num_active_tuples_ = 0;

  for( int k = 0; k < num_cameras_; k++ )
  {
    cam_models_[k] = cv_ext::PinholeCameraModel();
    images_points_[k].clear();
    r_mats_[k] = cv::Mat();
    t_vecs_[k] = cv::Mat();
  }

  tuples_masks_.clear();
  per_view_errors_.clear();
}