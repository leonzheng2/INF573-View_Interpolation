import cv2
import geometry
import numpy as np
import match


def keypointsMatching(left, right):
    """Returns the list of the matched keypoints from input images left and right."""
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(left, None)
    kpts2, desc2 = akaze.detectAndCompute(right, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    sortedmatches = sorted(matches, key=lambda x: x.distance)
    nb_matches = 300
    good_matches = sortedmatches[:nb_matches]
    obj = []
    scene = []

    for i in range(len(good_matches)):
        # -- Get the keypoints from the good matches
        obj.append(kpts1[good_matches[i].queryIdx].pt)
        scene.append(kpts2[good_matches[i].trainIdx].pt)

    F, mask = cv2.findFundamentalMat(np.array(obj), np.array(scene), cv2.FM_RANSAC)
    correct_kpts1 = []
    correct_kpts2 = []
    correct_matches = []

    for i in range(len(mask)):
        if mask[i, 0] > 0:
            correct_kpts1.append(obj[i])
            correct_kpts2.append(scene[i])
            correct_matches.append(good_matches[i])

    print(len(correct_kpts1))
    matches_image = np.empty((max(left.shape[0], right.shape[0]), left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(left, kpts1, right, kpts2, correct_matches, matches_image)
    cv2.waitKey(0)

    return correct_kpts1, correct_kpts2, matches_image, F

def rectification(left, right, correct_kpts1, correct_kpts2, F):
    """Returns rectified images using OpenCV rectification algorithm"""
    """
    We have implemented our rectification method based on papar
    Physically-valid view synthesis by image interpolation, Charles R. Dyer Steven M. Seitz.

    In this case, we can obtain the matrix of rotation, translation and scale to de-rectify, but
    this algorithm has a very poor performance compared to OpenCV built-in rectification method.
    Furthermore, this method is not good enough to calculate the disparity, so we comment the few following lines
    and we use OpenCV.
    """
    correct_kpts1 = np.array(correct_kpts1)
    correct_kpts1 = correct_kpts1.reshape((correct_kpts1.shape[0] * 2, 1))
    correct_kpts2 = np.array(correct_kpts2)
    correct_kpts2 = correct_kpts2.reshape((correct_kpts2.shape[0] * 2, 1))
    shape = (left.shape[1], left.shape[0])

    rectBool, H1, H2 = cv2.stereoRectifyUncalibrated(correct_kpts1, correct_kpts2, F, shape, threshold=1)
    R1 = cv2.warpPerspective(left, H1, shape)
    R2 = cv2.warpPerspective(right, H2, shape)

    return R1, R2

def stereoMatching(R1, R2, mi, ma):
    """Compute disparity mapping between two rectified images, using Kolmogorov and Zabih's graph cuts stereo matching algorithm."""
    """
    We found that the quality of disparity map obtained by OpenCV StereoSGBM is depend
    strongly the choice of parameters. So we implement the method based on paper:
    Kolmogorov and zabih’sgraph cuts stereo matching algorithm, Pauline Tan Vladimir Kolmogorov, Pascal Monasse.
    It suffice to set a good disparity range [Min, Max].
    Attention: with this python version implementation, this method is very slow, so to quickly have a result,
    we force here the images used can't be larger than 200*200
    """
    K = -1
    lambda_ = -1
    lambda1 = -1
    lambda2 = -1
    params = match.Parameters(is_L2=True,
                              denominator=1,
                              edgeThresh=8,
                              lambda1=lambda1,
                              lambda2=lambda2,
                              K=K,
                              maxIter=4)
    # create match instance
    is_color = True if R1.shape[-1] == 3 else False
    m = match.Match(R1, R2, is_color)
    m.SetDispRange(mi, ma)
    m = match.fix_parameters(m, params, K, lambda_, lambda1, lambda2)
    disparity = m.kolmogorov_zabih()

    return disparity

def displayDispImg(disparity, OCCLUDED=int(2 ** 31 - 1)):
    """Display the disparity result of the graph cut as an image. Blue color corresponds to OCCLUDED"""
    dispImg = np.zeros((disparity.shape[0], disparity.shape[1],3), dtype=np.uint8)
    dispImg[:, :, 0] = 255
    dispImg[:, :, 1] = 255
    dispImg[:, :, 2] = 0

    flat = disparity.flatten()
    print(flat)
    flat = np.unique(flat)
    print(flat)
    flat.sort()

    minVal = flat[0]
    maxVal = flat[-2]
    print(minVal, maxVal)

    for i in range(disparity.shape[0]):
        for j in range(disparity.shape[1]):
            if disparity[i,j] != OCCLUDED:
                c = 255 - 255 * (disparity[i,j] - minVal) / (maxVal - minVal)
                print(c)
                dispImg[i, j, 0] = c
                dispImg[i, j, 1] = c
                dispImg[i, j, 2] = c
    return dispImg

def viewInterpolation(imgLpath, imgRpath, mi, ma, resize, scale):
    """Using precedent methods to compute view interpolation"""
    print("Reading images...")
    left = cv2.imread(imgLpath)
    right = cv2.imread(imgRpath)

    print("Compute keypoints matching...")
    correct_kpts1, correct_kpts2, matches_image, F = keypointsMatching(left, right)
    cv2.imshow("key pointsmatching", matches_image)
    cv2.waitKey(0)

    print("Computing rectification...")
    R1, R2 = rectification(left, right, correct_kpts1, correct_kpts2, F)
    cv2.imshow("rectified_left", R1)
    cv2.imshow("rectified_right", R2)
    cv2.imwrite("./results/rectifiedL.png", R1)
    cv2.imwrite("./results/rectifiedR.png", R2)
    cv2.waitKey(0)

    if resize:
        print("Resizing pictures...")
        R1 = cv2.resize(R1, dsize=None, fx=scale, fy=scale)
        R2 = cv2.resize(R2, dsize=None, fx=scale, fy=scale)
        cv2.imwrite("./results/rectifiedL.png", R1)
        cv2.imwrite("./results/rectifiedR.png", R2)

    print("Computing disparity...")
    disparity = stereoMatching(R1, R2, mi, ma)
    np.save("./results/dispArray.npy", disparity)
    dispImg = displayDispImg(disparity)
    cv2.imshow("Disparity", dispImg)
    cv2.imwrite("./results/dispImage.png", dispImg)
    cv2.waitKey(0)

    print("Computing interpolation...")
    ir = geometry.interpolate(1.5, R1, R2, disparity)
    cv2.imshow("interpolated view", ir)
    cv2.waitKey(0)
    cv2.imwrite("./results/interpolated.png", ir)
    print("Save successfully interpolated view !")

if __name__ == '__main__':
    # Choosing images to rectify
    imgLpath = './images/perra_7.jpg'
    imgRpath = './images/perra_8.jpg'
    # Setting disparity range
    dispMin = -11
    dispMax = 7
    # If rescale is True, the size of rectified images will be multiplied by scale
    rescale = True
    scale = 0.25
    # Compute view interpolation.
    """Running time with these parameters shouldn't exceed 5 minutes."""
    viewInterpolation(imgLpath, imgRpath, dispMin, dispMax, rescale, scale)