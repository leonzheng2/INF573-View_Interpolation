# view-interpolation

This part of the view interpolation by Seitz and Dyer is coded in C++ and Python.

Python: main method is in view_interpolation.py
C++: main method is in interpolation.cpp. Execution file is called view_interpolation

Parameters are choosen to validate the interpolation method. We didn't apply rectification and derectification, because we are using OpenCV's StereoSGBM disparity matching algorithm.

One can test the algorithm on other pictures in the directory images/

Results are saved in the directory results/python/ for the python code, and in results/c++/ for the C++ code.
