#pragma once

#include "cv_ext/pinhole_camera_model.h"

#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>

#include <iostream>

#define GTEST_COUT   std::cout << "\033[0;32m" << "[          ] " << "\033[0;0m"

template <typename _T> void fillRandom( cv::Mat &m )
{
  srand(time(NULL));
  for( int r = 0; r < m.rows; r++ )
  {
    _T *r_ptr = m.ptr<_T>(r);
    for (int c = 0; c < m.cols * m.channels(); c++)
      *r_ptr++ = static_cast<_T>(rand());
  }
}

bool identicalMats(const cv::Mat &m1, const cv::Mat &m2 );
bool quasiIdenticalMats( const cv::Mat &m1, const cv::Mat &m2, double epsilon );


std::vector< cv::Point2f > generateCheckerboardImage( cv::Mat &cb_img, const cv_ext::PinholeCameraModel &cam_model,
                                                      const cv::Size &board_size, float square_len,
                                                      const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec,
                                                      bool has_white_circle = false );