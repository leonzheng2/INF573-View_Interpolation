# INF573 View-interpolation

INF573 Image Analysis (2018-2019, R. Keriven) course project done with Chen Jiabin at Ecole polytechnique.
Implementation of Seitz and Dyer view interpolation algorithm, using Kolmogorov and Zabih's graph cuts stereo matching algorithm.

Project report is in the Git repo.

Source codes are in the subdirectories. 

The basic-view-interpolation directory uses OpenCV StereoSGBM stereo matching algorithm, and don't work a lot on our own pictures.

The graph-cuts-linear-interpolation directory uses Kolmogorov and Zabih's graph cuts stereo matching algorithm. It is a bit slow.
