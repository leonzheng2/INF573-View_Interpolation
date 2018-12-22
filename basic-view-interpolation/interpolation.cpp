#include "image.h"
#include "interpolation.h"

using namespace std;
using namespace Eigen;
using namespace cv;


void Interpolation::matchingKeypoints(const Image<uchar>&I1, const Image<uchar>& I2, int max_good_matches, vector<Point2f>& kptsL, vector<Point2f>& kptsR, bool display) {
    //Finding keypoints
    Ptr<AKAZE> D = AKAZE::create();
    vector<KeyPoint> m1, m2;
    Mat desc1, desc2;
    D->detectAndCompute(I1, noArray(), m1, desc1);
    D->detectAndCompute(I2, noArray(), m2, desc2);
    if(display){
        //Displaying keypoints
        Mat J1;
        drawKeypoints(I1, m1, J1);
        imshow("I1", J1);
        Mat J2;
        drawKeypoints(I2, m2, J2);
        imshow("I2", J2);
        waitKey(0);
    }

    //For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one.
    BFMatcher matcher(NORM_HAMMING, true);
    vector<DMatch>  matches;
    matcher.match(desc1, desc2, matches);
    cout << matches.size() << " matches for BF Matching" << endl;
    if(display){
        //Displaying matches
        Mat res;
        drawMatches(I1, m1, I2, m2, matches, res);
        imshow("match", res);
        waitKey(0);
    }

    sort(matches.begin(), matches.end(), distance_for_matches);
    //-- Draw only "good" matches (first max_good_matches)
    vector<DMatch>  good_matches;
    for( int i = 0; i < max_good_matches; i++ )
    {
        good_matches.push_back( matches[i]);
    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( m1[ good_matches[i].queryIdx ].pt );
        scene.push_back( m2[ good_matches[i].trainIdx ].pt );
    }

    Mat mask;
    Mat F = findFundamentalMat(obj, scene, CV_FM_RANSAC, 3., 0.99, mask);
    cout << "Fundamental matrix: " << F << endl;
    vector<DMatch> correct_matches;
    for(int i = 0; i < mask.rows; i++){
        if( mask.at<uchar>(i, 0) > 0){
            kptsL.push_back(obj[i]);
            kptsR.push_back(scene[i]);
            correct_matches.push_back(matches[i]);
        }
    }
    cout << "Final number of keypoints matches: " << correct_matches.size() << endl;

    if(display){
        Mat res;
        drawMatches(I1, m1, I2, m2, correct_matches, res);
        imshow("correct match", res);
        waitKey(0);
    }

}

void Interpolation::rectify(const Image<uchar>&I1, const Image<uchar>& I2, const vector<Point2f>& kptsL, const vector<Point2f>& kptsR, Image<uchar>& R1, Image<uchar>& R2, RectParam &D) {

    // compute mean of x_value and y_value of key points in imgL
    double x_meanL = 0.;
    double y_meanL = 0.;
    for (auto &i : kptsL) {
        x_meanL += i.x;
        y_meanL += i.y;
    }
    // compute mean of x_value and y_value of key points in imgR
    double x_meanR = 0.;
    double y_meanR = 0.;
    for (auto &i : kptsR) {
        x_meanR += i.x;
        y_meanR += i.y;
    }
    x_meanL /= kptsL.size();
    y_meanL /= kptsL.size();
    x_meanR /= kptsR.size();
    y_meanR /= kptsR.size();

    // measurement matrix
    MatrixXd M(4, kptsL.size());
    for(int i=0; i<kptsL.size(); i++){
        M(0, i) = kptsL[i].x - x_meanL; // apply directly translation
        M(1, i) = kptsL[i].y - y_meanL;
        M(2, i) = kptsR[i].x - x_meanR;
        M(3, i) = kptsR[i].y - y_meanR;
    }

    // Singular value decomposition of M
    JacobiSVD<MatrixXd> svd( M, ComputeFullV | ComputeFullU );
    if(svd.computeU()){
        cout << "SVD is successful" << endl;
    } else {
        cout << "SVD isn't successful" << endl;
    }
    MatrixXd U = svd.matrixU();

    //    U_ = U[:, :3], U1 = U_[:2, :], U2 = U_[2:, :]
    MatrixXd U_ = U.leftCols(3);
    MatrixXd U1 = U_.topRows(2);
    MatrixXd U2 = U_.bottomRows(2);

    // A1 = U1[:2, :2], d1 = U1[:, 2], A2 = U2[:2, :2], d2 = U2[:, 2]
    MatrixXd A1 = U1.block(0,0,2,2);
    MatrixXd d1 = U1.col(2);
    MatrixXd A2 = U2.block(0,0,2,2);
    MatrixXd d2 = U2.col(2);

    //    define B_i, U_1' and U_2'
    Matrix3d B1;
    B1(2,2) = 1;
    B1(2,0) = 0;
    B1(2,1) = 0;
    B1.block(0,0,2,2) = A1.inverse();
    B1.block(0,2,2,1) = A1.inverse()*d1;

    Matrix3d B2;
    B2(2,2) = 1;
    B2(2,0) = 0;
    B2(2,1) = 0;
    B2.block(0,0,2,2) = A2.inverse();
    B2.block(0,2,2,1) = A2.inverse()*d2;

    MatrixXd U1_prime = U1*B2;
    MatrixXd U2_prime = U2*B1;

    double x1 = U1_prime(0, 2); double y1 = U1_prime(1, 2);
    double theta1 = atan(y1 / x1);

    double x2 = U1_prime(0, 2); double y2 = U1_prime(1, 2);
    double theta2 = atan(y2 / x2);


    Matrix2d rot1, rot2;
    rot1(0,0) = cos(theta1); rot1(0,1) = sin(theta1);
    rot1(1,0) = -sin(theta1);rot1(1,1) = cos(theta1);

    rot2(0,0) = cos(theta1); rot2(0,1) = sin(theta1);
    rot2(1,0) = -sin(theta1);rot2(1,1) = cos(theta1);


    Matrix3d B, B_inv;
    B.block(0,0,2,3) = rot1*U1_prime;
    B.bottomRows(1) = (rot2*U2_prime).topRows(1);
    if(B.determinant()!=0){
        B_inv = B.inverse();
    }else{
        B(2,0) = 0; B(2,1) = 0; B(2,2) = 1;
        B_inv = B.inverse();
    }

    MatrixXd tmp;
    tmp = rot2*(U2_prime*B_inv);
    double s = tmp(1,1);
    MatrixXd H_s(2,2);
    H_s(0,0) = 1;H_s(0,1) = 0;
    H_s(1,0) = 0;H_s(1,1) = 1. / s;


    Vector2d T_1(-x_meanL, -y_meanL), T_2(-x_meanR, -y_meanR);

    cout << "Finished to compute rotation, scale, translation matrix or vector" << endl;

    int rows1 = I1.rows, cols1 = I1.cols;
    MatrixXd map1_0(rows1, cols1), map1_1(rows1, cols1);
    Vector2d pos(0,0);
    for(int x = 0; x<cols1; x++){
        for(int y=0; y<rows1; y++){
            pos(0) = x; pos(1) = y;
            pos = pos + T_1;
            pos = rot1 * pos;
            map1_0(y, x) = pos(0);
            map1_1(y, x) = pos(1);
        }
    }
    double w_min1 = map1_0.minCoeff(), w_max1 = map1_0.maxCoeff();
    double h_min1 = map1_1.minCoeff(), h_max1 = map1_1.maxCoeff();


    map1_0 = (map1_0.array() - w_min1).matrix();
    map1_1 = (map1_1.array() - h_min1).matrix();

    int rectified_h1 = int(h_max1 - h_min1 + 1), rectified_w1 = int(w_max1 - w_min1 + 1);

    R1 = Image<uchar>(rectified_w1, rectified_h1);
    for(int x = 0; x<cols1; x++){
        for(int y=0; y<rows1; y++){
            R1(int(map1_1(y, x)), int(map1_0(y, x))) = I1(x, y);
        }
    }
    cout << "R1 builded" << endl;


    int rows2 = I2.rows, cols2 = I2.cols;
    MatrixXd map2_0(rows2, cols2), map2_1(rows2, cols2);
    for(int x = 0; x<cols2; x++){
        for(int y=0; y<rows2; y++){
            pos(0) = x; pos(1) = y;
            pos = pos + T_2;
            pos = rot2 * pos;
            map2_0(y, x) = pos(0);
            map2_1(y, x) = pos(1);
        }
    }

    map2_0 = (map2_0.array() - w_min1).matrix();
    map2_1 = (map2_1.array() - h_min1).matrix();

    cout << "Building R2" << endl;

    int vx, vy;
    R2 = Image<uchar>(rectified_w1, rectified_h1);
    for(int x = 0; x<cols2; x++){
        for(int y=0; y<rows2; y++){
            vx = int(map2_1(y, x)); vy = int(map1_0(y, x));
            if(vx>=0 && vy>=0 && vx<rectified_w1 && vy<rectified_h1){
                R2(vx, vy) = I2(x, y);
            }

        }
    }

    cout << "R2 builded" << endl;

    D.theta1 = theta1; D.theta2 = theta2;
    D.T1 = T_1; D.T2 = T_2; D.s = s;



}

void Interpolation::disparityMapping(const Image<uchar>& R1, const Image<uchar>& R2, Image<short>& disparity){
    //    min_disp = 0
    //    num_disp = 32
    //    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
    //                                   numDisparities=num_disp,
    //                                   blockSize=5,
    //                                   P1=200,
    //                                   P2=500,
    //                                   disp12MaxDiff=10,
    //                                   uniquenessRatio=10,
    //                                   speckleWindowSize=60,
    //                                   speckleRange=7
    //                                   )

    Ptr<StereoSGBM> sgbm_ = StereoSGBM::create();
    sgbm_->setBlockSize(5);
    sgbm_->setDisp12MaxDiff(10);
    sgbm_->setP1(200);
    sgbm_->setP2(500);
    sgbm_->setMinDisparity(0);
    sgbm_->setNumDisparities(32);
    sgbm_->setUniquenessRatio(10);
    sgbm_->setPreFilterCap(4);
    sgbm_->setSpeckleRange(7);
    sgbm_->setSpeckleWindowSize(60);

    sgbm_->compute(R1, R2, disparity);

    cout << R1.size << " is the matrix size of R1" << endl;
    cout << disparity.size << " is the matrix size of disparity" << endl;

    for(int x=0; x<disparity.width(); x++){
        for(int y=0; y<disparity.height(); y++){
            disparity(x,y) = short(disparity(x,y) / 16);
        }
    }
}

void Interpolation::interpolate(const Image<uchar>& R1, const Image<uchar>& R2, const Image<short>& disparity, Image<uchar>& IR, RectParam& D){
    IR = Image<uchar>(R1.width(), R1.height());
    for(int y=0; y<R1.height(); y++){
        for(int x1=0; x1<R1.width(); x1++){
            int x2 = x1 - disparity(x1,y);
            int x_i = int((2-D.i)*x1 + (D.i-1)*x2);
            if(x_i >=0 && x_i < IR.width()){
                IR(x_i, y) = uchar((2-D.i)*R1(x1,y) + (D.i-1)*R2(x2,y));
            } else {
                IR(x_i, y) = 0;
            }
        }
    }
}

void Interpolation::derectify(const Image<uchar>& IR, const RectParam &D, Image<uchar>& I){
    /* Compute derectifying rotation, scale and translation matrix */
    Vector2d T_i = (2-D.i)*D.T1 + (D.i-1)*D.T2;
    MatrixXd H_s_i(2,2);
    double s_i = (2-D.i)*1.0 + (D.i-1)*D.s;
    H_s_i(0,0) = 1;H_s_i(0,1) = 0;
    H_s_i(1,0) = 0;H_s_i(1,1) = 1. / D.s;
    Matrix2d rot_i;
    double theta_i = (2-D.i)*D.theta1 + (D.i-1)*D.theta2;
    rot_i(0,0) = cos(theta_i); rot_i(0,1) = -sin(theta_i);
    rot_i(1,0) = sin(theta_i);rot_i(1,1) = cos(theta_i);

    /* Compute derectification transformation */
    MatrixXd xCoordinates(IR.rows, IR.cols);
    MatrixXd yCoordinates(IR.rows, IR.cols);
    for(int x=0; x<IR.rows; x++){
        for(int y=0; y<IR.cols; y++){
            Vector2d pixel(x,y);
            Vector2d newPixel = rot_i*H_s_i*pixel - T_i;
            xCoordinates(x,y) = newPixel(0);
            yCoordinates(x,y) = newPixel(1);
        }
    }
    double xMin = xCoordinates.minCoeff();
    double xMax = xCoordinates.maxCoeff();
    double yMin = yCoordinates.minCoeff();
    double yMax = yCoordinates.maxCoeff();

    xCoordinates = (xCoordinates.array() - xMin).matrix();
    yCoordinates = (yCoordinates.array() - yMin).matrix();

    int height = int(round(yMax - yMin) + 1);
    int width = int(round(xMax - xMin) + 1);
    I = Image<uchar>(width, height);
    for(int x=0; x<IR.cols; x++){
        for(int y=0; y<IR.rows; y++){
            I(int(round(xCoordinates(y,x))), int(round(yCoordinates(y,x)))) = IR(x,y);
        }
    }
}

int main()
{
    /* Parameters */
    int max_good_matches = 300;
    double i = 1.5;
    bool not_rectified = false;

    /* Seitz and Dyer view interpolation */
    cout << "Reading left and right images..." << endl;
    Image<uchar> I1 = Image<uchar>(imread("../images/statueL.png", CV_LOAD_IMAGE_GRAYSCALE));
	Image<uchar> I2 = Image<uchar>(imread("../images/statueR.png", CV_LOAD_IMAGE_GRAYSCALE));

    Image<uchar> R1, R2;
    RectParam D;
    if(not_rectified){
        cout << "Finding keypoints and matching them..." << endl;
        vector<Point2f> kptsL, kptsR;
        Interpolation::matchingKeypoints(I1, I2, max_good_matches, kptsL, kptsR, false);

        cout << "Rectifying images..." << endl;
        Interpolation::rectify(I1, I2, kptsL, kptsR, R1, R2, D);
	} else {
	    R1 = I1;
	    R2 = I2;
	}
    imshow("left_rectified", R1);
    imwrite("../results/c++/leftRect.png", R1);
    imshow("right_rectified", R2);
    imwrite("../results/c++/rightRect.png", R2);
    waitKey(0);

    cout << "Computing disparity..." << endl;
    Image<short> disparity;
    Interpolation::disparityMapping(R1, R2, disparity);
    Image<uchar> dispImg = disparity.greyImage();
    imshow("disparity", dispImg);
    imwrite("../results/c++/disparity.png", dispImg);
    waitKey(0);

    cout << "Interpolating rectified intermediate view..." << endl;
    Image<uchar> IR;
    D.i = i;
    Interpolation::interpolate(R1, R2, disparity, IR, D);
    imshow("left_rectified-right_rectified", IR);
    imwrite("../results/c++/left+rightRect.png", dispImg);
    waitKey(0);

    if(not_rectified){
        cout << "Derectifying interpolated view..." << endl;
        Image<uchar> I;
        Interpolation::derectify(IR, D, I);
        imshow("left-right", I);
        imwrite("../results/c++/left+right.png", I);
        waitKey(0);
    }

	return 0;
}
