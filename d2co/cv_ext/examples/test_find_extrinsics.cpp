#include <iostream>
#include <sstream>
#include <string>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

#include "cv_ext/cv_ext.h"
#include "apps_utils.h"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

int main(int argc, char **argv)
{

  string app_name( argv[0] ), filelist_name, directory_name, calibration_filename, extrinsics_filename;
  bool show_corners = false;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "filelist_name,f", po::value<string > ( &filelist_name )->required(),
    "Input images list file name" )
  ( "directory_name,d", po::value<string > ( &directory_name ),
    "Input images directory" )
  ( "calibration_filename,c", po::value<string > ( &calibration_filename )->required(),
    "Input calibration filename" )
  ( "extrinsics_filename,e", po::value<string > ( &extrinsics_filename )->required(),
    "Output extrinsics basic filename" )
  ( "show_corners,s", "Show detected corners" );

  po::variables_map vm;
  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;
      return 0;
    }

    if ( vm.count ( "show_corners" ) )
      show_corners = true;


    po::notify ( vm );
  }
  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }
  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }

  cout << "Loading images filenames from file : "<<filelist_name<< endl;
  cout << "Loading camera parameters from file : "<<calibration_filename<< endl;
  cout << "Saving extrinsics to file : "<<extrinsics_filename<< endl;

  std::vector<std::string> images_names;
  if( !readFileNames ( filelist_name, images_names ) )
    return -1;

  if(!directory_name.empty())
    directory_name += "/";
  std::vector<std::string> image_paths (images_names.size(), directory_name);
  for(int i = 0; i < int(images_names.size()); i++ )
    image_paths[i] += images_names[i];

  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(calibration_filename);
  cv_ext::CameraCalibration calib;
  calib.setCamModel(cam_model);
  calib.setBoardSize(Size(8,6));
  calib.setSquareSize(0.03983333333333333333);
  calib.setPyramidNumLevels(2);
  
  for( auto &f : image_paths )
    calib.addImageFile (f);

  if( show_corners )
  {
    for( int i = 0; i < calib.numImages(); i++ )
    {
      Mat corners_img;
      calib.getCornersImage(i,corners_img,2);
      cv_ext::showImage(corners_img, "Corners");
    }
  }
  
  cv::Mat r_vec, t_vec;
  calib.computeAverageExtrinsicParameters( r_vec, t_vec );

  cv_ext::write3DTransf(extrinsics_filename, r_vec, t_vec );

  return 0;
}
