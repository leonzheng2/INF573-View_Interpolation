import cv2
import numpy as np

import matplotlib.pyplot as plt


def grayImage(img):
    maxVal = np.max(img)
    minVal = np.min(img)
    alpha = 255. / (maxVal - minVal)
    beta = -minVal * alpha
    dst = cv2.convertScaleAbs(src=img, dst=None, alpha=alpha, beta=beta)
    return dst


def disparity_map(img1, img2):
    window_size = 5
    min_disp = -16
    num_disp = 16*2
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )
    # stereo = cv2.StereoSGBM_create(-128, 128, 5, 600, 2400, -1, 4, 1, 150, 2, True)
    disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
    return disparity


def rectify(kpts1, kpts2, img1, img2):
    """
    kpts1: numpy array of coordonnees of key points in image1, shape (nb_points, 2)
    kpts2: numpy array of coordonnees of key points in image2, shape (nb_points, 2)
    img1: left gray image of shape (h, w)
    img2: right gray image of shape (h, w)

    """
    # change the reference in image1
    x_centroids1, y_centroids1 = np.mean(kpts1, axis=0)

    print(x_centroids1)
    print("...........")
    print(y_centroids1)

    T1 = np.array([-x_centroids1, -y_centroids1])
    kpts1 = kpts1 + T1
    # print(kpts1)

    # change the reference in image2
    x_centroids2, y_centroids2 = np.mean(kpts2, axis=0)

    T2 = np.array([-x_centroids2, -y_centroids2])
    kpts2 = kpts2 + T2

    # measurement matrix
    M = np.concatenate([kpts1.T, kpts2.T], axis=0)
    # print(M)
    # Singular value decomposition of M
    U, sigma, Vh = np.linalg.svd(M)

    U_ = U[:, :3]
    U1 = U_[:2, :]
    U2 = U_[2:, :]

    # partition U_i
    A1 = U1[:2, :2]
    d1 = U1[:, 2]
    A2 = U2[:2, :2]
    d2 = U2[:, 2]

    # define B_i, U_1' and U_2'
    B1 = np.zeros(shape=(3, 3))
    B1[-1, -1] = 1
    B1[:2, :2] = np.linalg.inv(A1)
    B1[:2, 2] = -np.dot(np.linalg.inv(A1), d1)

    B2 = np.zeros(shape=(3, 3))
    B2[-1, -1] = 1
    B2[:2, :2] = np.linalg.inv(A2)
    B2[:2, 2] = -np.dot(np.linalg.inv(A2), d2)

    U1_prime = np.dot(U1, B2)
    U2_prime = np.dot(U2, B1)

    # calculate theta1, theta2
    x1 = U1_prime[0, -1]
    y1 = U1_prime[1, -1]
    theta1 = np.arctan(y1 / x1)

    x2 = U2_prime[0, -1]
    y2 = U2_prime[1, -1]
    theta2 = np.arctan(y2 / x2)

    # rotation matrix
    R1 = np.array([[np.cos(theta1), np.sin(theta1)],
                   [-np.sin(theta1), np.cos(theta1)]])

    R2 = np.array([[np.cos(theta2), np.sin(theta2)],
                   [-np.sin(theta2), np.cos(theta2)]])

    # calculate B and B_inv
    B = np.zeros(shape=(3, 3))
    B[:2, :] = np.dot(R1, U1_prime)
    B[2, :] = np.dot(R2, U2_prime)[0, :]

    try:
        B_inv = np.linalg.inv(B)
    except LinAlgError:
        B[2, :] = np.array([0, 0, 1])
        B_inv = np.linalg.inv(B)

    # calculate s and H_s
    tmp = np.dot(R2, np.dot(U2_prime, B_inv))
    s = tmp[1, 1]

    H_s = np.array([[1, 0],
                    [0, 1. / s]])

    # rectify I1 and I2
    # create firstly a map between original position and rectified position
    rows1, cols1 = img1.shape
    map1 = np.zeros((rows1, cols1, 2))
    for h in range(rows1):
        for w in range(cols1):
            map1[h, w] = np.dot(R1, np.array([w, h]) + T1)

    w_min1 = np.min(map1[:, :, 0])
    w_max1 = np.max(map1[:, :, 0])
    h_min1 = np.min(map1[:, :, 1])
    h_max1 = np.max(map1[:, :, 1])
    map1[:, :, 0] = map1[:, :, 0] - w_min1
    map1[:, :, 1] = map1[:, :, 1] - h_min1
    rectified_h1 = int(round(h_max1 - h_min1) + 1)
    rectified_w1 = int(round(w_max1 - w_min1) + 1)
    rectified1 = np.zeros((rectified_h1, rectified_w1))
    for h in range(rows1):
        for w in range(cols1):
            rectified1[int(round(map1[h, w, 1])), int(round(map1[h, w, 0]))] = img1[h, w]

    rows2, cols2 = img2.shape
    map2 = np.zeros((rows2, cols2, 2))
    for h in range(rows2):
        for w in range(cols2):
            map2[h, w] = np.dot(H_s, np.dot(R2, np.array([w, h]) + T2))

    map2[:, :, 0] = map2[:, :, 0] - w_min1
    map2[:, :, 1] = map2[:, :, 1] - h_min1

    rectified2 = np.zeros_like(rectified1)
    for h in range(rows2):
        for w in range(cols2):
            y = int(round(map2[h, w, 1]))
            x = int(round(map2[h, w, 0]))
            if 0 <= y < rectified_h1 and 0 <= x < rectified_w1:
                rectified2[y, x] = img2[h, w]

    return rectified1.astype(np.uint8), rectified2.astype(np.uint8), theta1, theta2, s, T1, T2


def interpolate(i, imgL, imgR, disparity):
    """
    :param i:
    :param imgL:
    :param imgR:
    :param disparity:
    :return:
    """
    ir = np.zeros_like(imgL)
    for y in range(imgL.shape[0]):
        for x1 in range(imgL.shape[1]):
            x2 = int(x1 - disparity[y, x1])
            x_i = int((2 - i) * x1 + (i - 1) * x2)
            if 0 <= x_i < ir.shape[1] and 0 <= x2 < imgR.shape[1]:
                ir[y, x_i] = int((2 - i) * imgL[y, x1] + (i - 1) * imgR[y, x2])
    return ir.astype(np.uint8)


def deRectify(ir, theta1, theta2, T1, T2, s, i):
    """
    :param ir: numpy array, interpolated image to be de-rectified
    :param theta1: float, rotation angle in left image
    :param theta2: float, rotation angle in right image
    :param T1: numpy array, translation vector in left image
    :param T2: numpy array, translation vector in right image
    :param s:  float number, scale factor
    :param i:  float number
    :return: numpy array, de-rectified image
    """
    theta_i = (2 - i) * theta1 + (i - 1) * theta2
    s_i = (2 - i) * 1. + (i - 1) * s
    T_i = (2 - i) * T1 + (i - 1) * T2
    H_s_i = np.array([[1, 0],
                      [0, s_i]])
    R_i = np.array([[np.cos(theta_i), -np.sin(theta_i)],
                    [np.sin(theta_i), np.cos(theta_i)]])
    # de-rectify
    rows, cols = ir.shape
    mapping = np.zeros((rows, cols, 2))
    for h in range(rows):
        for w in range(cols):
            mapping[h, w] = np.dot(R_i, np.dot(H_s_i, np.array([w, h]))) - T_i

    w_min = np.min(mapping[:, :, 0])
    w_max = np.max(mapping[:, :, 0])
    h_min = np.min(mapping[:, :, 1])
    h_max = np.max(mapping[:, :, 1])
    mapping[:, :, 0] = mapping[:, :, 0] - w_min
    mapping[:, :, 1] = mapping[:, :, 1] - h_min
    de_rectified_h = int(round(h_max - h_min) + 1)
    de_rectified_w = int(round(w_max - w_min) + 1)
    de_rectified = np.zeros((de_rectified_h, de_rectified_w))
    for h in range(rows):
        for w in range(cols):
            de_rectified[int(round(mapping[h, w, 1])), int(round(mapping[h, w, 0]))] = ir[h, w]

    return de_rectified.astype(np.uint8)


def main(imgLpath, imgRpath, nb_matches, i, not_rectified):
    print("Reading images...")
    left = cv2.imread(imgLpath, 0)
    cv2.imshow("left", left)
    right = cv2.imread(imgRpath, 0)
    cv2.imshow("right", right)
    cv2.waitKey(0)

    if not_rectified:
        print("Compute keypoints matching...")
        akaze = cv2.AKAZE_create()
        # akaze = cv2.xfeatures2d.SIFT_create()
        kpts1, desc1 = akaze.detectAndCompute(left, None)
        kpts2, desc2 = akaze.detectAndCompute(right, None)
        matcher = cv2.BFMatcher(cv2.NORM_L2, True)
        matches = matcher.match(desc1, desc2)
        sortedmatches = sorted(matches, key=lambda x: x.distance)
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

        res = np.empty((max(left.shape[0], right.shape[0]), left.shape[1] + right.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(left, kpts1, right, kpts2, correct_matches, res)
        cv2.imwrite("./results/python/keypoints.jpg", res)
        cv2.imshow("keypoints", res)
        cv2.drawMatches(left, kpts1, right, kpts2, good_matches, res)
        cv2.imwrite("./results/python/keypoints_without_RANSAC.jpg", res)
        cv2.waitKey(0)

        print("Computing rectification...")
        # grey1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        # grey2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        dst1, dst2, theta1, theta2, s, T1, T2 = rectify(np.array(obj), np.array(scene), left, right)
    else:
        dst1 = left
        dst2 = right
    cv2.imwrite("./results/python/rectify1.png", dst1)
    cv2.imwrite("./results/python/rectify2.png", dst2)
    cv2.imshow("left_rectified", dst1)
    cv2.imshow("right_rectified", dst2)
    cv2.waitKey(0)

    print("Computing disparity...")
    disparity = disparity_map(dst1, dst2)
    disparity_img = grayImage(disparity)
    cv2.imwrite("./results/python/disparity.png", disparity_img)
    print(disparity)
    cv2.imshow("disparity", disparity_img)
    cv2.waitKey(0)

    print("Computing interpolation...")
    ir = interpolate(i, dst1, dst2, disparity)
    cv2.imwrite("./results/python/interpolated.png", ir)
    cv2.imshow("interpolated_rectified", ir)
    print(ir)
    cv2.waitKey(0)

    if not_rectified:
        print("Computing derectification...")
        de_rect = deRectify(ir, theta1, theta2, T1, T2, s, i)
        cv2.imwrite("./results/python/derectified.png", de_rect)
        cv2.imshow("interpolated", de_rect)
        cv2.waitKey(0)


if __name__ == '__main__':
    #Choose images
    imgL = './images/statueL.png'
    imgR = './images/statueR.png'
    #Choose i between 1 and 2 to select to position of the intermediate view
    i = 1.5
    #Choose max number of keypoints matchings
    nMax = 150
    #Choose to apply of not the rectification/derectification steps to the input images: select True if images aren't already rectified
    not_rectified = False

    """View interpolation"""
    main(imgL, imgR, nMax, i, not_rectified)
