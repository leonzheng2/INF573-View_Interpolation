CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(view-interpolation)
include_directories( /usr/local/Cellar/eigen/3.3.7/include/eigen3 )

ADD_EXECUTABLE(view_interpolation interpolation.cpp interpolation.h image.cpp image.h Eigen/Core Eigen/SVD Eigen/Dense)
TARGET_LINK_LIBRARIES(view_interpolation ${OpenCV_LIBS})
