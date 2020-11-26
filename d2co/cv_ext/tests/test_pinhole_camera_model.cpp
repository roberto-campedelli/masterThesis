#include "cv_ext/pinhole_camera_model.h"

#include <gtest/gtest.h>


TEST (PinholeCameraModelTest, FilePersistenceTest)
{
  cv::Mat_<double> k(3,3), dist_coeff(8,1);

  k.setTo(0);

  k(0,0) = 999.99;
  k(1,1) = 888.88;
  k(0,2) = 320.11;
  k(1,2) = 199.99;

  double val = 0.0;
  for( int i = 0; i < 8; i++, val += 0.11 )
    dist_coeff(i,0) = val;

  cv_ext::PinholeCameraModel cm1(k,640,480,dist_coeff), cm2;

  cm1.writeToFile( "cam_test.yml" );
  cm2.readFromFile( "cam_test.yml" );
  ASSERT_TRUE(cm1 == cm2);
}

TEST (PinholeCameraModelTest, YamlNodePersistenceTest)
{
  cv::Mat_<double> k0(3,3), dist_coeff0(8,1);
  cv::Mat_<double> k1(3,3), dist_coeff1(8,1);

  k0.setTo(0);
  k1.setTo(0);

  k0(0,0) = 999.99;
  k0(1,1) = 888.88;
  k0(0,2) = 320.11;
  k0(1,2) = 199.99;
  k1(0,0) = 1999.99;
  k1(1,1) = 1888.88;
  k1(0,2) = 1320.11;
  k1(1,2) = 1199.99;

  double val0 = 0.0;
  for( int i = 0; i < 8; i++, val0 += 0.11 )
    dist_coeff0(i,0) = val0;
  double val1 = 0.0;
  for( int i = 0; i < 8; i++, val1 += 0.21 )
    dist_coeff1(i,0) = val1;

  cv_ext::PinholeCameraModel cam0(k0,640,480, dist_coeff0);
  cv_ext::PinholeCameraModel cam1(k1,640,480, dist_coeff1);

  // load cam params into yaml nodes
  YAML::Node root, node_cam0, node_cam1;
  cam0.write(node_cam0);
  cam1.write(node_cam1);
  root["camera0"] = node_cam0;
  root["camera1"] = node_cam1;

  std::ofstream out_file("test_many_cams.yml");
  out_file << root;
  out_file.close();

  // load cams from file
  cv_ext::PinholeCameraModel out_cam0, out_cam1;
  YAML::Node new_root = YAML::LoadFile("test_many_cams.yml");
  out_cam0.read(new_root["camera0"]);
  out_cam1.read(new_root["camera1"]);

  bool assert0 = cam0 == out_cam0;
  bool assert1 = cam1 == out_cam1;
  ASSERT_TRUE(assert0 && assert1);
}