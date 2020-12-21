#include <cstdio>
#include <string>
#include <sstream>
#include <time.h>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "raster_object_model2D.h"

#include "apps_utils.h"

/* TODO
 * -Update render with RoI
 */
#define TEST_DEPTH_MAP 0

/*
"0_apple
 1_ball
 2_banana
 3_battery
 4_bottle
 5_can
 6_chair
 7_guitar
 8_lighter
 9_mug
 10_shoe
 11_sofa
 12_tv
*/

#define category_id 0

namespace po = boost::program_options;
using namespace std;
using namespace cv;


static void quickGuide()
{
  cout << "Use the keyboard to move the object model:" << endl<< endl;
  
  objectPoseControlsHelp();
  
  cout << "Visulization modes:"  << endl;
  cout << "[1] to change mode between points and segments mode" << endl;
  cout << "[2] to enable/disable the visulization of the normals" << endl;
  cout << "[3] to enable/disable the visulization of the object mask" << endl;
  cout << "[4] to enable/disable the visulization of the object depth map" << endl;
  cout << "[5] to enable/disable the visulization of the object render (only available if the model has been loaded with \"color\" option)" << endl;
  cout << "[v] to enable/disable the visulization of the object vertices" << endl;
  cout << "[r] to enable/disable the visulization of the object reference frame" << endl;
  cout << "[b] to enable/disable the visulization of the object bounding box" << endl;;
  cout << "[g] to change the light position in a new random position" << endl;
}


void applyTfToPoints(cv::Mat& r_vec, cv::Mat& t_vec, std::vector<Point3f>& input_pts, std::vector<Point3f>& output_pts)
{
  output_pts.resize(input_pts.size());
  for (int i=0; i<input_pts.size(); ++i)
  {
    Point3f pt = input_pts[i];
    cv::Mat homogeneous_pt = (Mat_<float>(4,1) << pt.x, pt.y, pt.z, 1.0f);
    cv::Mat tf_mat;
    cv_ext::exp2TransfMat<float>(r_vec, t_vec, tf_mat);
    cv::Mat transformed_pt = tf_mat * homogeneous_pt;
    output_pts[i] = Point3f(transformed_pt.at<float>(0,0),
                            transformed_pt.at<float>(1,0),
                            transformed_pt.at<float>(2,0));
  }
}

template<typename T> T indexOfTheMin(vector<T> v){
  T tmp = v[0];
  int index = 0;
  for( int i=0; i< v.size(); i++){
    if(v[i] <= tmp){
      tmp = v[i];
      index = i;
    }
  }
  return index;
}

template<typename T> T indexOfTheMax(vector<T> v){
  T tmp = v[0];
  int index = 0;
  for( int i=0; i< v.size(); i++){
    if(v[i] >= tmp){
      tmp = v[i];
      index = i;
    }
  }
  return index;
}

 int getTopLeftPointIndex(vector<Point2f> point_vector){
  vector<float> x_vector(8);
  vector<float> y_vector(8);
  for (unsigned int i = 0; i < point_vector.size(); i++)
  { x_vector[i] = point_vector[i].x;
    y_vector[i] = point_vector[i].y;  
  }
  int index;
  vector<int> top4index(4);
  for(int i =0; i < 4; i++){
    top4index[i] = indexOfTheMin(y_vector);
    y_vector[top4index[i]] = 2000;
  }
  float minx = 2000;
  for (int i = 0; i < 4; i++)
    {
      if(point_vector[top4index[i]].x < minx ){
          minx = point_vector[top4index[i]].x;
          index = top4index[i];
      }
    }
      
  return index;
}

int getTopRightPointIndex(vector<Point2f> point_vector){
  vector<float> x_vector(8);
  vector<float> y_vector(8);
  for (unsigned int i = 0; i < point_vector.size(); i++)
  { x_vector[i] = point_vector[i].x;
    y_vector[i] = point_vector[i].y;  
  }
  int index;
  vector<int> top4index(4);
  for(int i =0; i < 4; i++){
    top4index[i] = indexOfTheMin(y_vector);
    y_vector[top4index[i]] = 2000;
  }
  float maxx = 0;
  for (int i = 0; i < 4; i++)
    {
      if(point_vector[top4index[i]].x > maxx ){
          maxx = point_vector[top4index[i]].x;
          index = top4index[i];
      }
    }
      
  return index;
}

int getBottomLeftPointIndex(vector<Point2f> point_vector){
  vector<float> x_vector(8);
  vector<float> y_vector(8);
  for (unsigned int i = 0; i < point_vector.size(); i++)
  { x_vector[i] = point_vector[i].x;
    y_vector[i] = point_vector[i].y;  
  }  int index;
  vector<int> top4index(4);
  for(int i =0; i < 4; i++){
    top4index[i] = indexOfTheMax(y_vector);
    y_vector[top4index[i]] = 0;
  }
  float minx = 2000;
  for (int i = 0; i < 4; i++)
    {
      if(point_vector[top4index[i]].x < minx ){
          minx = point_vector[top4index[i]].x;
          index = top4index[i];
      }
    }
      
  return index;
}

//get depth and length --> prendi i 4 punti più in alto (y minore) e prendi più profondo e più vicino(max z e min z)

void getLengthAndDepth(vector<Point3f> point3d, float &length_m, float &depth){
  vector<float> x_vector(8);
  vector<float> y_vector(8);
  for (unsigned int i = 0; i < point3d.size(); i++)
  { x_vector[i] = point3d[i].x;
    y_vector[i] = point3d[i].y;  
  }
  int index;
  vector<int> top4index(4);
  for(int i =0; i < 4; i++){
    top4index[i] = indexOfTheMin(y_vector);
    y_vector[top4index[i]] = 2000;
  }
  int deepest_point_index = top4index[0];
  int closest_point_index = top4index[0];
  for(int i = 1; i < top4index.size(); i++){
    if(point3d[top4index[i]].z >= point3d[deepest_point_index].z)
      deepest_point_index = top4index[i];
    if(point3d[top4index[i]].z < point3d[closest_point_index].z)
      closest_point_index = top4index[i];
  }

  //cout << " deepest and closest point : [ " << deepest_point_index << " , " << closest_point_index << " ]" << endl; 


  //length_m = sqrt(pow(point3d[deepest_point_index].z - point3d[closest_point_index].z, 2) + pow(point3d[deepest_point_index].y - point3d[closest_point_index].y, 2));
  length_m = point3d[deepest_point_index].z - point3d[closest_point_index].z;
  depth = point3d[closest_point_index].z + length_m/2;
}


void printLabel(string image_path, cv::Mat calib, int cat_id,Vec4f bbox,Vec3f dim, Point3f location, double rotation_y, double alpha){

  boost::filesystem::path pathImg(image_path);
  string file_name = pathImg.filename().string();

  string image_id = file_name.substr(0, file_name.rfind("."));

  ofstream file;
  ofstream file2;
  //file.open(str.append(file_name, ".txt");
  file.open("images_label.json", ofstream::app);
  file2.open("annotations_label.json", ofstream::app);
  file << "\n\n{\"file_name\": \""<< file_name << "\"" 
        << ", \"id\": " << image_id
        << ", \"calib\": [" << calib.row(0) << ", " << calib.row(1) << ", " << calib.row(2) << "]}, \n"
        << endl;
  
  file2 <<"\n{\"image_id\": " << image_id
        <<", \"id\": " << image_id
        <<", \"category_id\": "<< cat_id
        <<", \"dim\": " << dim 
        <<", \"bbox\": " << bbox 
        <<", \"depth\": " << location.z
        <<", \"alpha\": " << alpha
        <<", \"truncated\": " << 0
        <<", \"occluded\": " << 0
        <<", \"location\": " << location
        <<", \"rotation_y\": " << rotation_y
       << "}," << endl;
  file.close();
  file2.close();


  cout << "\n{\"file_name\": \""<< file_name << "\"" 
        << ", \"id\": " << image_id
        << ", \"calib\": [" << calib.row(0) << ", " << calib.row(1) << ", " << calib.row(2) << "]}, \n"

        <<"\n\n{\"image_id\": " << image_id
        <<", \"id\":" << image_id
        <<", \"category_id\": "<< cat_id
        <<", \"dim\": " << dim 
        <<", \"bbox\": " << bbox 
        <<", \"depth\": " << location.z
        <<", \"alpha\": " << alpha
        <<", \"truncated\": " << 0
        <<", \"occluded\": " << 0
        <<", \"location\": " << location
        <<", \"rotation_y\": " << rotation_y
       << "}," << endl;

}


int main(int argc, char **argv)
{
  // Initialize random seed
  srand (time(NULL));
  
  string app_name( argv[0] ), model_filename, camera_filename, image_filename;
  double scale_factor = 1.0;
  int top_boundary = -1, bottom_boundary = -1, left_boundary = -1, rigth_boundary = -1;
  string rgb_color_str;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &model_filename )->required(),
    "STL, PLY, OBJ, ...  model file" )
  ( "camera_filename,c", po::value<string > ( &camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "image_filename,i", po::value<string > ( &image_filename ),
    "Optional background image file" )
  ( "scale_factor,s", po::value<double > ( &scale_factor ),
    "Optional scale factor" )
  ( "tb", po::value<int> ( &top_boundary ),
    "Optional region of interest: top boundary " )
  ( "bb", po::value<int> ( &bottom_boundary ),
    "Optional region of interest: bottom boundary " )
  ( "lb", po::value<int> ( &left_boundary ),
    "Optional region of interest: left boundary " )  
  ( "rb", po::value<int> ( &rigth_boundary ),
    "Optional region of interest: rigth boundary" )
  ( "color", "Try to load also model colors" )
  ( "rgb", po::value<string> ( &rgb_color_str ),
    "Unifrom RGB color (in HEX format) to be used to render the model. The --rgb and --color options are mutually exclusive" )
  ( "light", "Enable lighting in model rendering" );

  po::variables_map vm;
  bool has_color = false, has_light = false;
  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;
      return 0;
    }

    if ( vm.count ( "color" ) || vm.count ( "rgb" ) )
    {
      has_color = true;
      if ( vm.count ( "light" ) )
        has_light = true;
        
    }

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
              << endl << endl<<desc;
    return -1;
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
              << endl << endl<<desc;
    return -1;
  }
  
  Mat r_vec = (Mat_<double>(3,1) << 0,0,0),
      t_vec = (Mat_<double>(3,1) << 0,0,1.0);
  cv::Point3f point_light_pos(1,0,0), light_dir(0,0,-1);
  
  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(camera_filename);
  cam_model.setSizeScaleFactor(scale_factor);
  int img_w = cam_model.imgWidth(), img_h = cam_model.imgHeight();
  
  bool has_roi = false;
  cv::Rect roi;
  
  if( top_boundary != -1 || bottom_boundary != -1 || 
      left_boundary != -1 || rigth_boundary != -1 )
  {
    Point tl(0,0), br(img_w, img_h);
    
    if( top_boundary != -1 ) tl.y = top_boundary;
    if( left_boundary != -1 ) tl.x = left_boundary;
    if( bottom_boundary != -1 ) br.y = bottom_boundary;
    if( rigth_boundary != -1 ) br.x = rigth_boundary;
    
    has_roi = true;
    roi = cv::Rect(tl, br);
    cam_model.setRegionOfInterest(roi);
    cam_model.enableRegionOfInterest(true);
    roi = cam_model.regionOfInterest();
  }
   
  cout << "Loading model from file : "<<model_filename<< endl;
  cout << "Loading camera parameters from file : "<<camera_filename<< endl;
  if( !image_filename.empty() )
    cout << "Loading background image from file : "<<image_filename<< endl;
  cout << "Scale factor : "<<scale_factor<< endl;
  if(has_roi)
    cout << "Region of interest : "<<roi<< endl;
  
  RasterObjectModel3D obj_model;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.001);
  obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);
  if( has_color )
  {
    if( rgb_color_str.length() )
    {
      uint32_t rgb_color;
      std::stringstream ss;
      ss << std::hex << rgb_color_str;
      ss >> rgb_color;
      
      uchar r = (rgb_color&0XFF0000)>>16, g = (rgb_color&0XFF00)>>8, b = (rgb_color&0XFF);
      obj_model.setVerticesColor(cv::Scalar(r,g,b));
    }
    else
    {
      obj_model.requestVertexColors();
    }
    if( has_light )
      obj_model.requestRenderLighting();
  }

  if(!obj_model.setModelFile( model_filename ))
    return -1;

  // obj_model.setMinSegmentsLen(0.01);
  obj_model.computeRaster();

  has_color = obj_model.vertexColorsEnabled();

  cv::Mat background_img;
  if(!image_filename.empty())
  {
    background_img = cv::imread(image_filename, cv::IMREAD_COLOR);
    cv::resize(background_img, background_img, Size(img_w, img_h));
  }
  
  vector<Point2f> raster_pts, middle_points, vertices;
  vector<Vec4f> raster_segs;
  vector<float> raster_normals_dirs;

  bool segment_mode = false, show_normals = false,
       show_mask = false, show_depth = false, show_render = false,
       draw_vertices = false, draw_bb = false, draw_axis = false,
       print_info = false;

  quickGuide();

  bool exit_now = false;
  while( !exit_now )
  {      
    obj_model.setModelView(r_vec, t_vec);
    
    Mat background, draw_img;
    if( background_img.empty() )
      background = Mat( Size(img_w,img_h),
                        DataType<Vec3b>::type, CV_RGB( 0,0,0));
    else
      background = background_img.clone();

    if( has_roi )
      draw_img = background(roi);
    else
      draw_img = background;
      
    if( show_mask )
    {
      draw_img.setTo(cv::Scalar(255,255,255), obj_model.getMask());
    }
    else if( show_depth )
    {
      Mat depth_img_f = obj_model.getModelDepthMap(), depth_img;
  
#if TEST_DEPTH_MAP
      
      vector<Point3f> pts = obj_model.getPoints();
      obj_model.projectRasterPoints( raster_pts );
      int useful_pts = 0;
      double avg_abs_diff = 0, avg_diff = 0, max_abs_diff = 0;
      int m_r, m_c;
      for( int i = 0; i < raster_pts.size(); i++ )
      {
        int r = cvRound(raster_pts[i].y), c = cvRound(raster_pts[i].x);
        float d = depth_img_f.at<float>(r,c);

        if( d != -1 )
        {
          
          double  scene_pt[] = {pts[i].x, pts[i].y, pts[i].z }, transf_pt[3];
          // ceres::AngleAxisRotatePoint uses the rodriguez formula only away from zero
          ceres::AngleAxisRotatePoint((double *)r_vec.data, scene_pt, transf_pt);
          float z = transf_pt[2] + t_vec.at<double>(2);

          float diff = d-z;
          avg_diff += diff;
//           cout<<"("<<r<<","<<c<<") : |" <<d<<" - "<<z<<" | = "<<diff<<endl;
          
          diff = fabs(diff);
          if( diff > max_abs_diff )
          {
            max_abs_diff = diff;
            m_r = r;
            m_c = c;
          }
          avg_abs_diff += diff;
          useful_pts++;
        }
      }
      
      depth_img_f.at<float>(m_r,m_c) = 1.0;
      cout<<"Average abs depth diff : "<<avg_abs_diff/useful_pts<<"Average depth diff : "
          <<avg_diff/useful_pts<<" Max abs depth diff  "<<max_abs_diff<<endl;
      
#endif
      
      depth_img_f *= 255;
      depth_img_f.convertTo(depth_img,cv::DataType<uchar>::type );
      cv::cvtColor(depth_img, depth_img, cv::COLOR_GRAY2BGR);
      
      if( background_img.empty() )
        depth_img.copyTo(draw_img);
      else
        depth_img.copyTo(draw_img, obj_model.getMask());
    }
    else if( show_render )
    {
      if( background_img.empty() )
        obj_model.getRenderedModel().copyTo(draw_img);
      else
      {
        Mat render_img = obj_model.getRenderedModel();
        render_img.copyTo(draw_img, obj_model.getMask());
      }
    }
    else
    {
      if( segment_mode )
      {
        if( show_normals )
        {
          obj_model.projectRasterSegments( raster_segs, raster_normals_dirs );

          middle_points.clear();
          middle_points.reserve(raster_segs.size());
          for(int i = 0; i < int(raster_segs.size()); i++)
          {
            Vec4f &seg = raster_segs[i];
            middle_points.push_back(Point2f((seg[0] + seg[2])/2, (seg[1] + seg[3])/2 ));
          }
          cv_ext::drawSegments( draw_img, raster_segs );
          cv_ext::drawNormalsDirections(draw_img, middle_points, raster_normals_dirs );
        }
        else
        {
          obj_model.projectRasterSegments( raster_segs );
          cv_ext::drawSegments( draw_img, raster_segs );
        }
      }
      else
      {
        if( show_normals )
        {
          obj_model.projectRasterPoints( raster_pts, raster_normals_dirs);
          cv_ext::drawNormalsDirections(draw_img, raster_pts, raster_normals_dirs );
        }
        else
        {
          obj_model.projectRasterPoints( raster_pts );
          cv_ext::drawPoints(draw_img, raster_pts );
        }
      }
    }

    if( draw_vertices )
    {
      obj_model.projectVertices( vertices );
      cv_ext::drawCircles(draw_img, vertices, 1, Scalar(255, 0, 255) );
    }

    if( draw_bb )
    {
      vector< Vec4f > proj_bb_segs;
      vector< Point2f > proj_bb_pts;
      obj_model.projectBoundingBox ( proj_bb_segs );
      obj_model.projectBoundingBox ( proj_bb_pts );

      cv_ext::drawSegments( draw_img, proj_bb_segs, Scalar(0, 255, 255) );
      cv_ext::drawCircles(draw_img, proj_bb_pts, 2, Scalar(0, 0, 255) );

      // retrieve bbox 3d points in obj ref. frame
      std::vector<Point3f> bb_v, bb_v_transformed;
      obj_model.getBoundingBox().vertices(bb_v);
      
      float width_model = obj_model.getBoundingBox().width;
      float height_model =obj_model.getBoundingBox().height;
      float depth_model = obj_model.getBoundingBox().depth;
      
      cout << "width model = " << width_model << endl;
      cout << "height model = " << height_model << endl;
      cout << "length model = " << depth_model << endl; 
      
      // transform 3dbbox points into camera frame
      applyTfToPoints(r_vec, t_vec, bb_v, bb_v_transformed);
      
      // print 3d vertices in camera frame
      /*
      cout << "\n3DBBox vertices in camera frame:" << endl;
      for (int i=0; i<bb_v_transformed.size(); ++i)
      {
        cout << "v[" << i << "]: " <<  bb_v_transformed[i].x << " " << bb_v_transformed[i].y << " " << bb_v_transformed[i].z << endl;
      }
      cout << endl;

*/
     /* 
      cout << "\n3DBBox vertices in obj frame:" << endl;
      for (int i=0; i<bb_v_transformed.size(); ++i)
      {
        cout << "v[" << i << "]: " <<  bb_v[i].x << " " << bb_v[i].y << " " << bb_v[i].z << endl;
      }
      cout << endl;
      */
      
      // print 3d vertices in image frame
      /*
      cout << "3DBBox vertices in image coordinates " << endl;
      for(int i = 0; i < proj_bb_pts.size(); i++){      
        cout << "p[" << i << "]: " << proj_bb_pts[i].x << " " << proj_bb_pts[i].y << endl;
      }
      cout << endl;
      */


      //2D bounding box of object in the image (0-based index):
      //contains left, top, right, bottom pixel coordinates
      //topLeft point = (lowest x, lowest y)
      //bottomLeft point = (lowest x, highest y)
      //topRight point  =  (highest x, lowest y)
      
      /*   
      vector<float> proj_bb_pts_x (8);
      vector<float> proj_bb_pts_y (8);

      for (unsigned int i = 0; i < proj_bb_pts.size(); i++)
      { proj_bb_pts_x[i] = proj_bb_pts[i].x;
        proj_bb_pts_y[i] = proj_bb_pts[i].y;  
      }
      
      for (int i = 0; i < proj_bb_pts.size(); i++){
      cout << " v["<<i<<"].x =" << proj_bb_pts_x[i] << endl;
      }
      
      for (int i = 0; i < proj_bb_pts.size(); i++){
      cout << " v["<<i<<"].y =" << proj_bb_pts_y[i] << endl;
      }
      */
      
      //among the 4 point with the lowest y i take the lowest x (tofleft) and the highest x (bottomleft)
      //among the 4 point with the lowest x i take the highest y (topRight)
   
      int topLeft = getTopLeftPointIndex(proj_bb_pts);
      int bottomLeft = getBottomLeftPointIndex(proj_bb_pts);
      int topRight = getTopRightPointIndex(proj_bb_pts);

      cout << "//////////////\ntop left point [blue] : " << topLeft << endl;
      cout << "bottom left point [black] : " << bottomLeft << endl;
      cout << "top right point [green] :" << topRight << endl;
      cout << "/////////////////" << endl;

      float width_px = sqrt(pow(proj_bb_pts[topRight].x - proj_bb_pts[topLeft].x, 2) +  pow(proj_bb_pts[topRight].y - proj_bb_pts[topLeft].y, 2));
      float height_px = sqrt(pow(proj_bb_pts[bottomLeft].y - proj_bb_pts[topLeft].y, 2) + pow(proj_bb_pts[bottomLeft].x - proj_bb_pts[topLeft].x, 2));

      //cout << " width px " << width_px << endl;
      //cout << "height px " << height_px << endl;

      const cv::Vec4f bb2d ( proj_bb_pts[topLeft].x, proj_bb_pts[topLeft].y, width_px, height_px); 

      cout << "\n2dbbox coordinates(topLeft x, topleft y, width, height) = " << bb2d << endl;

      //cout << "\n points 2d\n" << proj_bb_pts << endl;


    

      Point2f center2d;
      center2d.x = proj_bb_pts[topLeft].x + width_px/2.0;
      center2d.y = proj_bb_pts[topLeft].y + height_px/2.0;
      //i draw blue point on top left and on the center
      vector<Point2f> blue_points;
      blue_points.push_back(proj_bb_pts[topLeft]);
      blue_points.push_back(center2d);
      cv_ext::drawCircles(draw_img, blue_points, 2, Scalar(255, 0, 0) );

      vector<Point2f> green_points;
      green_points.push_back(proj_bb_pts[topRight]);
      cv_ext::drawCircles(draw_img, green_points, 2, Scalar(0, 255, 0) );

      vector<Point2f> black_points;
      black_points.push_back(proj_bb_pts[bottomLeft]);
      cv_ext::drawCircles(draw_img, black_points, 2, Scalar(0, 0, 0) );
      


      //box dimension in meters
      float width_m = sqrt(pow(bb_v_transformed[topRight].x - bb_v_transformed[topLeft].x, 2) +  pow(bb_v_transformed[topRight].y - bb_v_transformed[topLeft].y, 2));
      float height_m = sqrt(pow(bb_v_transformed[bottomLeft].y - bb_v_transformed[topLeft].y, 2) + pow(bb_v_transformed[bottomLeft].x - bb_v_transformed[topLeft].x, 2));

      float length_m;

      float depth;

      //i retrieve length and depth in meters
      getLengthAndDepth(bb_v_transformed, length_m, depth);

      cv::Vec3f dim (height_m, width_m, length_m);    
      cout << "\n3dbb dimension(h, w, d) = " << dim << endl;

      //3D object center location x,y,z in camera coordinates (in meters)

      Point3f center3bbox;
      center3bbox.x = bb_v_transformed[topLeft].x + width_m/2.0;
      center3bbox.y = bb_v_transformed[topLeft].y + height_m/2.0;
      center3bbox.z = depth;
      //cv_ext::drawCircles(draw_img, center3bbox, 2, Scalar(0, 0, 255) );

      cout << "\ncenter 3bbox = " << center3bbox << endl;

      //get camera matrix
      cv::Mat cameraMatrix = cam_model.cameraMatrix();
      //cout << "\ncamera matrix = " << cameraMatrix << endl;

      //get tf matrix
      cv::Mat tf_mat;
      cv_ext::exp2TransfMat<float>(r_vec, t_vec, tf_mat);

      //get rotation matrix
      Mat r_mat;
      cv_ext::angleAxis2RotMat<float>(r_vec, r_mat);
    
      //cout << "r_vec = " << r_vec << endl;
      //cout << "t_vec = " << t_vec << endl;
      //cout << "tf_mat = " << tf_mat << endl;
      //cout << "r_mat = " << r_mat << endl;


      // rotation on axis y
      double rotation_y = r_vec.at<double>(1,0);
      //cout << "\nrotation y = " << rotation_y << endl;

      //normalize rotation_y in [-pi, pi]
      double normalizedRotation_y = rotation_y;
      if (normalizedRotation_y > CV_PI)
         normalizedRotation_y -= 2 * CV_PI;
      if (normalizedRotation_y < - CV_PI)
         normalizedRotation_y+= 2 * CV_PI;
      cout << "normalized rotation y = " << normalizedRotation_y << endl;  


      // calculation of the projection matrix
      //Mat small_tf_mat;
      //tf_mat(Range(0, tf_mat.rows -1), Range(0, tf_mat.cols)).copyTo(small_tf_mat);
      //small_tf_mat.convertTo(small_tf_mat, 6);
      //cout << "small_tf_mat = " << small_tf_mat << endl;

      //Mat projectMatrix = cameraMatrix*small_tf_mat;

      Mat projectMatrix;
      Mat zeroVector = (Mat_<double>(3,1) << 0, 0, 0);
      hconcat(cameraMatrix, zeroVector, projectMatrix);
      //cout << "\nprojection Matrix = " << projectMatrix << endl;

      //calculation of alpha - object angle
      float x_center_pixel = proj_bb_pts[0].x + (proj_bb_pts[1].x - proj_bb_pts[0].x)/2;
      double alpha = double(rotation_y - atan2(double(x_center_pixel) - cameraMatrix.at<double>(0,2), cameraMatrix.at<double>(0,0)));
      if (alpha > CV_PI)
        alpha -= 2 * CV_PI;
      if (alpha < - CV_PI)
        alpha += 2 * CV_PI;
      cout << "\nalpha = " << alpha << endl;

      cout << endl;
      cout << endl;

      //cout << "categories = " << "0_apple - 1_ball - 2_banana - 3_battery - 4_bottle - 5_can - 6_chair -"
      //                         << "7_guitar - 8_lighter - 9_mug -10_shoe -11_sofa -12_tv" << endl;
 
 
     
      //print info for labeling 
      if(print_info){

        //string category_id;
        //cout << "Insert category = " << endl;
        //cin >> category_id;
        printLabel(image_filename, projectMatrix, category_id ,bb2d,  dim, center3bbox, normalizedRotation_y, alpha );

      }   
      
  }

    
    
    if( draw_axis )
    {
      vector< Vec4f > proj_segs_x, proj_segs_y,  proj_segs_z;
      obj_model.projectAxes ( proj_segs_x, proj_segs_y, proj_segs_z );

      cv_ext::drawSegments( draw_img, proj_segs_x, Scalar(0, 0, 255) );
      cv_ext::drawSegments( draw_img, proj_segs_y, Scalar(0, 255, 0) );
      cv_ext::drawSegments( draw_img, proj_segs_z, Scalar(255, 0, 0) );
    }
    
    if( has_roi )
    {
      cv::Point dbg_tl = roi.tl(), dbg_br = roi.br();
      dbg_tl.x -= 1; dbg_tl.y -= 1;
      dbg_br.x += 1; dbg_br.y += 1;
      cv::rectangle( background, dbg_tl, dbg_br, cv::Scalar(255,255,255));
    }
    
    imshow("Test model", background);
    int key = cv_ext::waitKeyboard();
      
    parseObjectPoseControls( key, r_vec, t_vec );

    switch(key)
    {
      case '1':
        segment_mode = !segment_mode;
        break;
      case '2':
        show_normals = !show_normals;
        break;
      case '3':
        show_mask = !show_mask;
        if( show_mask )
          show_depth = show_render = false;
        break;
      case '4':
        show_depth = !show_depth;
        if( show_depth )
          show_mask = show_render = false;
        break;
      case '5':
        if( has_color )
        {
          show_render = !show_render;
          if( show_render )
            show_depth = show_mask = false;
        }
        break;
      case 'v' :
        draw_vertices = !draw_vertices;
        break;
      case 'b' :
        draw_bb = !draw_bb;
        break;
      case 'y' :
        print_info = !print_info;
      case 'r':
        draw_axis = !draw_axis;
        break;
      case 'g':
        point_light_pos.x = rand() - RAND_MAX/2;
        point_light_pos.y = rand() - RAND_MAX/2;
        point_light_pos.z = rand() - RAND_MAX/2;

        light_dir.x = rand() - RAND_MAX/2;
        light_dir.y = rand() - RAND_MAX/2;
        light_dir.z = rand() - RAND_MAX/2;
        
        if(point_light_pos.x || point_light_pos.y || point_light_pos.z )
          point_light_pos /= cv_ext::norm3D(point_light_pos);
        else
          point_light_pos = cv::Point3f(1,0,0);
        
        if(light_dir.x || light_dir.y || light_dir.z )
          light_dir /= cv_ext::norm3D(light_dir);
        else
          light_dir = cv::Point3f(0,0,-1);
        
        obj_model.setPointLightPos(point_light_pos);
        obj_model.setLightDirection(light_dir);
        break;
        
      case cv_ext::KEY_ESCAPE:
        exit_now = true;
        break;
    }
  }

  return 0;
}

