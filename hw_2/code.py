import cv2
import numpy as np


def read_image_grayscale(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def load_intrinsic_matrix(file_path):
    return np.loadtxt(file_path, delimiter=',')


def initialize_sift(number_of_octaves, contrast_threshold, edge_threshold, gaussian_sigma):
    # initializing Sift algorithm with the best params we could find by trials and error on reference image I was given.
    return cv2.SIFT_create(
        nOctaveLayers=number_of_octaves,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=gaussian_sigma
    )


def detect_keypoints_and_descriptors(sift, image):
    return sift.detectAndCompute(image, None)


def draw_keypoints(image, keypoints):
    # drawing the keypoints in image 1 and image 2
    image_keypoints = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(image_keypoints, (x, y), 2, (0, 0, 255), -1)
    return image_keypoints



def match_descriptors(descriptors1, descriptors2):
    # matching descriptors in image 1 and image 2 using the cv2.BFMatcher.match
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # matches = bf.match(descriptors1, descriptors2)
    #
    # # sorting the matches from the best match to the worst match
    # return sorted(matches, key=lambda x: x.distance)

    # matching descriptors in image 1 and image 2 using the cv2.BFMatcher.knnMatch and Threshold ratio of nearest to the 2'nd nearest descriptor
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    thresholded_matches = []
    for m, n in matches:
        if m.distance < 0.77 * n.distance:
            thresholded_matches.append(m)

    return thresholded_matches


def draw_matching_keypoints(image1, image2, keypoints1, keypoints2, matches):
    # drawing the matching keypoints in image 1 and image 2
    img1_matching_keypoints = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    img2_matching_keypoints = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    for m in matches:
        x1, y1 = int(keypoints1[m.queryIdx].pt[0]), int(keypoints1[m.queryIdx].pt[1])
        x2, y2 = int(keypoints2[m.trainIdx].pt[0]), int(keypoints2[m.trainIdx].pt[1])
        cv2.circle(img1_matching_keypoints, (x1, y1), 2, (0, 0, 255), -1)
        cv2.circle(img2_matching_keypoints, (x2, y2), 2, (0, 0, 255), -1)
    return img1_matching_keypoints, img2_matching_keypoints


def draw_first_n_matches_and_lines(image1, image2, keypoints1, keypoints2, matches, n=70):
    # drawing n random matching keypoints in image 1 and image 2, and also draws the lines between them
    img1_matches = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB)
    img2_matches = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2RGB)
    random_indices = np.random.choice(len(matches), size=min(len(matches), n), replace=False)

    for i in random_indices:
        match = matches[i]
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        cv2.circle(img1_matches, (int(pt1[0]), int(pt1[1])), 4, (0, 0, 255), 1)
        cv2.circle(img2_matches, (int(pt2[0]), int(pt2[1])), 4, (0, 255, 0), 1)
        cv2.line(img1_matches, (int(pt1[0]), int(pt1[1])), (int(pt2[0] + image1.shape[1]), int(pt2[1])), (0, 255, 255), 1)
        cv2.line(img2_matches, (int(pt2[0]), int(pt2[1])), (int(pt1[0] - image1.shape[1]), int(pt1[1])), (0, 255, 255), 1)
    return img1_matches, img2_matches


def compute_essential_matrix(points1, points2, K):
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC)
    return E, mask


def compute_fundamental_matrix(E, K):
    return np.linalg.inv(K).T @ E @ np.linalg.inv(K)


def draw_epipolar_lines(img, lines, pts):
    # drawing the epipolar lines on the given image
    height, width = img.shape
    img1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    curr = 10
    for height, pt1 in zip(lines, pts):
        color = (curr, (2 * curr) % 256, (3.5 * curr) % 256)
        curr = (curr + 10) % 256
        x0, y0 = map(int, [0, -height[2] / height[1]])
        x1, y1 = map(int, [width, -(height[2] + height[0] * width) / height[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1, tuple(pt1), 5, (0, 255, 0), 1)

    return img1


def main():
    #################### Reading the data ####################

    img1 = read_image_grayscale('example_3/I1.png')
    img2 = read_image_grayscale('example_3/I2.png')
    K = load_intrinsic_matrix('example_3/K.txt')



    #################### Section 1 ####################

    sift = initialize_sift(6, 0.041, 10, 1.8)

    keypoints1, descriptors1 = detect_keypoints_and_descriptors(sift, img1)
    keypoints2, descriptors2 = detect_keypoints_and_descriptors(sift, img2)

    img1_keypoints = draw_keypoints(img1, keypoints1)
    img2_keypoints = draw_keypoints(img2, keypoints2)

    cv2.imshow('Image 1 - Keypoints', img1_keypoints)
    cv2.imshow('Image 2 - Keypoints', img2_keypoints)

    # cv2.imwrite('Image_1_Keypoints.png', img1_keypoints)
    # cv2.imwrite('Image_2_Keypoints.png', img2_keypoints)


    #################### Section 2 ####################

    matches = match_descriptors(descriptors1, descriptors2)

    img1_matching_keypoints, img2_matching_keypoints = draw_matching_keypoints(img1, img2, keypoints1, keypoints2, matches)
    img1_matches, img2_matches = draw_first_n_matches_and_lines(img1, img2, keypoints1, keypoints2, matches)

    cv2.imshow('Image 1 - Matches Keypoints Before Filtering', img1_matching_keypoints)
    cv2.imshow('Image 2 - Matches Keypoints Before Filtering', img2_matching_keypoints)
    cv2.imshow('Matching Points and Lines 1 Before Filtering', img1_matches)
    cv2.imshow('Matching Points and Lines 2 Before Filtering', img2_matches)

    # cv2.imwrite('Image1_Matches_Keypoints_Before_Filtering.png', img1_matching_keypoints)
    # cv2.imwrite('Image2_Matches_Keypoints_Before_Filtering.png', img2_matching_keypoints)
    # cv2.imwrite('Matching_Points_And_Lines1_Before_Filtering.png', img1_matches)
    # cv2.imwrite('Matching_Points_And_Lines2_Before_Filtering.png', img2_matches)


    #################### Section 3 ####################

    matches_points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    matches_points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)

    E, mask = compute_essential_matrix(matches_points1, matches_points2, K)
    F = compute_fundamental_matrix(E, K)

    inlier_matches = [matches[i] for i in np.where(mask.ravel() == 1)[0]]
    points1_inliers = matches_points1[mask.ravel() == 1]
    points2_inliers = matches_points2[mask.ravel() == 1]

    img1_matches_after, img2_matches_after = draw_first_n_matches_and_lines(img1, img2, keypoints1, keypoints2,
                                                                            inlier_matches)

    # selecting random 70 matches subset
    random_indices = np.random.choice(len(points1_inliers), size=min(len(points1_inliers), 70), replace=False)
    sub_inlier_matches1 = points1_inliers[random_indices]
    sub_inlier_matches2 = points2_inliers[random_indices]

    lines1 = cv2.computeCorrespondEpilines(sub_inlier_matches2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(sub_inlier_matches1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    img1_epipolar_lines = draw_epipolar_lines(img1, lines1, sub_inlier_matches1.astype(int))
    img2_epipolar_lines = draw_epipolar_lines(img2, lines2, sub_inlier_matches2.astype(int))

    print("Fundamental Matrix (F):\n ", F)
    print()
    print("Essential Matrix (E):\n ", E)

    cv2.imshow('Matching Points and Lines 1 After Filtering', img1_matches_after)
    cv2.imshow('Matching Points and Lines 2 After Filtering', img2_matches_after)
    cv2.imshow('Image 1 with Epipolar Lines', img1_epipolar_lines)
    cv2.imshow('Image 2 with Epipolar Lines', img2_epipolar_lines)

    # cv2.imwrite('Matching_Points_and_Lines_1_After_Filtering.png', img1_matches_after)
    # cv2.imwrite('Matching_Points_and_Lines_2_After_Filtering.png', img2_matches_after)
    # cv2.imwrite('Image_1_with_Epipolar_Lines.png', img1_epipolar_lines)
    # cv2.imwrite('Image_2_with_Epipolar_Lines.png', img2_epipolar_lines)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
