# view_interpolation-graph_cuts_stereo_matching
Implementation of Seitz and Dyer view interpolation algorithm, using Kolmogorov and Zabih's graph cuts stereo matching algorithm.

Details are in the subdirectories. 

The basic-view-interpolation directory uses OpenCV StereoSGBM stereo matching algorithm, and don't work a lot on our own pictures.

The graph-cuts-linear-interpolation directory uses Kolmogorov and Zabih's graph cuts stereo matching algorithm. It is a bit slow.
