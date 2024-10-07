# Computer Vision Projects
This repository contains two assignments completed as part of a Computer Vision course. The projects showcase fundamental computer vision techniques and algorithms.

## Assignment 1: Triangle Detection and Classification
In this assignment, the task is to detect triangles in an image and classify them into different types based on their geometric properties. The process involves:

- Detecting Triangles: Using edge detection and contour-finding techniques to locate the triangles present in the image.

- Classification: Once the triangles are detected, they are sorted into three categories:
  - Equilateral Triangles: Triangles where all three sides are equal.
  - Isosceles Triangles: Triangles that have two equal sides.
  - Right-Angled Triangles: Triangles that have a 90-degree angle.

The goal is to analyze the image, identify all triangles, and then sort them into the appropriate geometric classification based on their angles and side lengths. This assignment combines basic image processing techniques with geometric shape analysis, providing hands-on experience in both fields.

## Assignment 2: Feature Detection and Matching
In this assignment, the focus is on more advanced techniques:

- Keypoint Detection: Identifying distinct points in an image using feature detectors like SIFT (Scale-Invariant Feature Transform), SURF (Speeded-Up Robust Features), or ORB (Oriented FAST and Rotated BRIEF). These keypoints serve as distinctive points in the image that can be tracked or matched across multiple images.

- Feature Matching: After detecting keypoints, the next step is to match them across different images. This is particularly useful in applications like image stitching, where multiple images are aligned to create a panorama.

- Image Registration: Aligning different images by matching detected features, enabling tasks such as object recognition, panorama generation, or 3D reconstruction.
This assignment demonstrates how keypoint detection and feature matching are foundational in many computer vision tasks that require comparing or combining images.
