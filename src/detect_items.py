import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms
from tqdm import tqdm


def distance(coords: tuple[int, int, int, int]) -> float:
    """
    Calculates the L2 distance between two points.

    Parameters
    ----------
    coords : tuple[int, int, int, int]
        The coordinates of the two points.

    Returns
    -------
    float
        The L2 distance between the two points.
    """
    x1, y1, x2, y2 = coords
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def angle_degrees(coords: tuple[int, int, int, int]) -> float:
    """
    Calculates the angle with the x-axis of the line segment between two points in degrees.

    Parameters
    ----------
    coords : tuple[int, int, int, int]
        The coordinates of the two points.

    Returns
    -------
    float
        The angle between the two points in degrees.
    """
    x1, y1, x2, y2 = coords
    return (np.arctan2(y2 - y1, x2 - x1) % np.pi) * 180 / np.pi


def skeletonize(mask_original: np.ndarray) -> np.ndarray:
    """
    Skeletonizes a binary mask.

    Parameters
    ----------
    mask_original : np.ndarray
        The binary mask to skeletonize.

    Returns
    -------
    np.ndarray
        The skeletonized mask.
    """
    mask = mask_original.copy()
    size = np.size(mask)
    skel = np.zeros(mask.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        mask = eroded.copy()

        zeros = size - cv2.countNonZero(mask)
        if zeros == size:
            done = True

    return skel


def extend_line(coords: tuple[int, int, int, int], img_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """
    Extends a line segment to the full image boundaries.

    Parameters
    ----------
    coords : tuple[int, int, int, int]
        The coordinates of the line segment.
    img_shape : tuple[int, ...]
        The shape of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The endpoints of the extended line.
    """
    x1, y1, x2, y2 = coords
    img_height = img_shape[0]
    img_width = img_shape[1]

    if x1 == x2:
        # Vertical line
        return (x1, 0, x2, img_height-1)
    elif y1 == y2:
        # Horizontal line
        return (0, y1, img_width-1, y2)
    else:
        # Compute slope and intercept
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Compute y at left and right image boundaries
        x_min, x_max = 0, img_width - 1
        y_min_2, y_max_2 = 0, img_height - 1
        x_min_2 = int((y_min_2 - b) / m)
        x_max_2 = int((y_max_2 - b) / m)

        x_min_3 = max(x_min, x_min_2)
        x_max_3 = min(x_max, x_max_2)
        y_min_3 = int(m * x_min_3 + b)
        y_max_3 = int(m * x_max_3 + b)

        return (x_min_3, y_min_3, x_max_3, y_max_3)


def cluster_lines_by_endpoints(
    line_list: list[tuple[int, int, int, int]],
    image_shape: tuple[int, ...],
    distance_threshold: float = 10.0,
) -> list[dict[str, Any]]:
    """
    Clusters lines by the closeness of their endpoints.

    Parameters
    ----------
    line_list : list[tuple[int, int, int, int]]
        The list of line segments.
    image_shape : tuple[int, ...]
        The shape of the image.
    distance_threshold : float, optional
        The threshold for the distance between the endpoints of the line segments.
        Existing lines are considered to be in the same cluster if the distance between their endpoints is less than this threshold.

    Returns
    -------
    list[dict[str, Any]]
        The list of clusters. Each cluster is a dictionary with the following keys:
        - "mean" : tuple[int, int, int, int]
            The mean of the endpoints of the line segments in the cluster.
        - "mean_angle" : float
            The mean angle of the line segments in the cluster.
        - "line" : list[tuple[int, int, int, int]]
            The list of line segments in the cluster.
        - "line_lengths" : list[float]
            The list of lengths of the line segments in the cluster.
    """
    line_list_extended = [(extend_line(line, image_shape), distance(line)) for line in line_list]
    clusters = []

    for line, line_length in line_list_extended:
        x1, y1, x2, y2 = line
        found = False

        for cluster in clusters:
            x1c, y1c, x2c, y2c = cluster["mean"]

            if distance((x1c, y1c, x1, y1)) < distance_threshold and distance((x2c, y2c, x2, y2)) < distance_threshold:
                cluster["line"].append(line)
                cluster["line_lengths"].append(line_length)

                line_and_lengths = list(zip(cluster["line"], cluster["line_lengths"]))

                x1_mean = sum([line[0] * ll for line, ll in line_and_lengths]) / sum(cluster["line_lengths"])
                y1_mean = sum([line[1] * ll for line, ll in line_and_lengths]) / sum(cluster["line_lengths"])
                x2_mean = sum([line[2] * ll for line, ll in line_and_lengths]) / sum(cluster["line_lengths"])
                y2_mean = sum([line[3] * ll for line, ll in line_and_lengths]) / sum(cluster["line_lengths"])

                cluster["mean"] = (int(x1_mean), int(y1_mean), int(x2_mean), int(y2_mean))
                cluster["mean_angle"] = angle_degrees(cluster["mean"])

                found = True
                break

        if not found:
            clusters.append({
                "mean": (x1, y1, x2, y2),
                "mean_angle": angle_degrees((x1, y1, x2, y2)),
                "line": [line],
                "line_lengths": [line_length],
            })

    return clusters


def select_clusters_by_angle(
    clusters: list[dict[str, Any]],
    angle_threshold: float = 12.0,
    cluster_pick_threshold: float = 45.0,
) -> list[list[dict[str, Any]]]:
    """
    Selects two clusters that are ~90 degrees apart and one is the most common angle.

    Parameters
    ----------
    clusters : list[dict[str, Any]]
        The list of clusters.
    angle_threshold : float, optional
        The threshold for the angle between the clusters.
    cluster_pick_threshold : float, optional
        The threshold for the angle between the clusters to be picked.

    Returns
    -------
    list[list[dict[str, Any]]]
        A list of two lists of clusters:
        - Close-to-horizontal clusters
        - Close-to-vertical clusters
    """
    angle_clusters = []

    for i, cluster in enumerate(clusters):
        found = False

        for a_c in angle_clusters:
            if abs(cluster["mean_angle"] - a_c["mean_angle"]) < angle_threshold:
                a_c["mean_angle"] = (a_c["mean_angle"] * len(a_c["indices"]) + cluster["mean_angle"]) / (len(a_c["indices"]) + 1)
                a_c["indices"].append(i)
                found = True
                break

        if not found:
            angle_clusters.append({"mean_angle": cluster["mean_angle"], "indices": [i]})

    angle_clusters.sort(key=lambda x: len(x["indices"]), reverse=True)

    selected_clusters = [angle_clusters[0]]
    for cluster in angle_clusters[1:]:
        if (
            abs(cluster["mean_angle"] - selected_clusters[0]["mean_angle"]) > cluster_pick_threshold
            and abs(abs(cluster["mean_angle"] - selected_clusters[0]["mean_angle"]) - 180) > cluster_pick_threshold
        ):
            selected_clusters.append(cluster)
            break

    selected_clusters.sort(key=lambda x: min(x["mean_angle"], 180 - x["mean_angle"]))

    return [[clusters[i] for i in a_c["indices"]] for a_c in selected_clusters]


def to_homogeneous(p: tuple[int, int]) -> np.ndarray:
    """
    Converts a 2D point to homogeneous coordinates.

    Parameters
    ----------
    p : tuple[int, int]
        The 2D point to convert to homogeneous coordinates.

    Returns
    -------
    np.ndarray
        The homogeneous coordinates of the point.
    """
    return np.array([p[0], p[1], 1.0])


def line_from_points(p1: tuple[int, int], p2: tuple[int, int]) -> np.ndarray:
    """
    Calculates the homogeneous line equation from two points.

    Parameters
    ----------
    p1 : tuple[int, int]
        The first point.
    p2 : tuple[int, int]
        The second point.

    Returns
    -------
    np.ndarray
        The homogeneous line equation.
    """
    return np.cross(to_homogeneous(p1), to_homogeneous(p2))


def draw_lines_on_image(
    image: np.ndarray,
    lines: list[np.ndarray],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draws lines on an image from a list of homogeneous line equations [a, b, c].

    Parameters
    ----------
    image : np.ndarray
        The image to draw the lines on.
    lines : list[np.ndarray]
        The list of homogeneous line equations [a, b, c] to draw on the image.
    color : tuple[int, int, int], optional
        The color of the lines.
    thickness : int, optional
        The thickness of the lines.

    Returns
    -------
    np.ndarray
        The image with the lines drawn on it.
    """
    h, w = image.shape[:2]
    img = image.copy()

    for line in lines:
        a, b, c = line
        if abs(b) > abs(a):
            y0, y1 = 0, h
            x0 = int((-c - b * y0) / a) if abs(a) > 1e-6 else 0
            x1 = int((-c - b * y1) / a) if abs(a) > 1e-6 else 0
        else:
            x0, x1 = 0, w
            y0 = int((-c - a * x0) / b) if abs(b) > 1e-6 else 0
            y1 = int((-c - a * x1) / b) if abs(b) > 1e-6 else 0

        # clip values to integer limits
        x0 = max(-1000000, min(x0, 1000000))
        y0 = max(-1000000, min(y0, 1000000))
        x1 = max(-1000000, min(x1, 1000000))
        y1 = max(-1000000, min(y1, 1000000))

        cv2.line(img, (x0, y0), (x1, y1), color, thickness)
    return img


def ransac_vanishing_point(
    lines: list[np.ndarray],
    num_iters: int = 1000,
    threshold: float = 0.005,
) -> np.ndarray | None:
    """
    Estimates the vanishing point of a set of lines using RANSAC.

    Parameters
    ----------
    lines : list[np.ndarray]
        The list of homogeneous line equations [a, b, c].
    num_iters : int, optional
        The number of iterations to run RANSAC.
    threshold : float, optional
        The threshold for the error of a line to be considered an inlier.

    Returns
    -------
    np.ndarray | None
        The estimated vanishing point, or None if no vanishing point was found.
    """
    best_vp = None
    max_inliers = 0

    if len(lines) < 2:
        return None

    for _ in range(num_iters):
        random_indices = np.random.choice(len(lines), 2, replace=False)
        l1, l2 = lines[random_indices[0]], lines[random_indices[1]]
        vp = np.cross(l1, l2)
        if np.abs(vp[2]) < 1e-6:
            continue
        vp /= vp[2]

        inliers = 0
        for l in lines:
            error = np.abs(np.dot(l, vp)) / (np.linalg.norm(l[:2]) * np.linalg.norm(vp))
            if error < threshold:
                inliers += 1

        if inliers > max_inliers:
            best_vp = vp.copy()
            max_inliers = inliers

    return best_vp


def estimate_rectifying_homography(vp1: np.ndarray | None, vp2: np.ndarray | None) -> np.ndarray:
    """
    Estimates the rectifying homography between two vanishing points.

    Parameters
    ----------
    vp1 : np.ndarray | None
        The first vanishing point.
    vp2 : np.ndarray | None
        The second vanishing point.

    Returns
    -------
    np.ndarray
        The rectifying homography matrix. Identity matrix if vanishing points are not found.
    """
    if vp1 is None or vp2 is None or abs(vp1[2]) < 1e-6 or abs(vp2[2]) < 1e-6:
        return np.eye(3)

    # Normalize vanishing points
    vp1 = vp1 / vp1[2]
    vp2 = vp2 / vp2[2]

    # Step 1: Projective rectification
    l_inf = np.cross(vp1, vp2)
    l_inf = l_inf / l_inf[2]
    H_proj = np.eye(3)
    H_proj[2] = l_inf

    # Step 2: Apply H_proj to vanishing points
    vp1_rect = H_proj @ vp1
    vp2_rect = H_proj @ vp2

    # Step 3: Rotation to align vp1_rect with x-axis
    vec = vp1_rect[:2]
    angle = -np.arctan2(vec[1], vec[0])
    angle = angle if angle > 0 else angle + 2 * np.pi
    angle = angle % np.pi
    angle = angle if angle < np.pi / 2 else angle - np.pi

    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

    H_rot = R @ H_proj

    # Step 4: Apply rotation to vanishing points
    vp1_r = R @ vp1_rect
    vp2_r = R @ vp2_rect

    # Step 5: Affine rectification — solve for b_hor
    x1, y1 = vp1_r[:2]
    x2, y2 = vp2_r[:2]

    A = x1 * x2
    B = x1 * y2 + x2 * y1
    C = x1 * x2 + y1 * y2

    if abs(A) < 1e-8:
        # Linear case: avoid divide by zero
        if abs(B) < 1e-8:
            b_hor = 0.0
        else:
            b_hor = -C / B
    else:
        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            b_hor = 0.0  # fallback if no real solution
        else:
            sqrt_disc = np.sqrt(discriminant)
            b1 = (-B + sqrt_disc) / (2 * A)
            b2 = (-B - sqrt_disc) / (2 * A)

            # Return smaller absolute value (less distortion)
            b_hor = b1 if abs(b1) < abs(b2) else b2

    H_aff = np.array([
        [1.0, b_hor, 0],
        [0.0, 1.0,   0],
        [0.0, 0.0,   1]
    ])

    # Final homography
    H_full = H_aff @ H_rot
    return H_full


def adjust_homography_to_view(image_shape: tuple[int, ...], H: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Adjusts a homography to the view of the image.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        The shape of the image.
    H : np.ndarray
        The homography matrix to adjust.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int]]
        The adjusted homography matrix and the new size of the image.
    """
    h, w = image_shape[:2]
    # Corners of original image
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners
    warped_corners = cv2.perspectiveTransform(corners, H)

    # Get bounding box
    x_coords = warped_corners[:, 0, 0]
    y_coords = warped_corners[:, 0, 1]

    min_x = np.floor(x_coords.min()).astype(int)
    min_y = np.floor(y_coords.min()).astype(int)
    max_x = np.ceil(x_coords.max()).astype(int)
    max_y = np.ceil(y_coords.max()).astype(int)

    # Compute translation to shift image into positive space
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)

    # Adjusted homography
    H_adjusted = T @ H

    new_size = (max_x - min_x, max_y - min_y)
    return H_adjusted, new_size


def lines_to_homogeneous(line_segments: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    """
    Converts line segments to homogeneous line equations.

    Parameters
    ----------
    line_segments : list[tuple[int, int, int, int]]
        The list of line segments.

    Returns
    -------
    list[np.ndarray]
        The list of homogeneous line equations.
    """
    return [line_from_points((x1, y1), (x2, y2)) for x1, y1, x2, y2 in line_segments]


def extract_potential_grid_line_clusters(
    image: np.ndarray,
    hough_kwargs: dict[str, Any] | None = None,
    do_skeletonize: bool = True,
    gaussian_size: int = 5,
) -> list[list[dict[str, Any]]]:
    """
    Extracts potential grid line clusters from an image.

    Parameters
    ----------
    image : np.ndarray
        The image to extract potential grid line clusters from.
    hough_kwargs : dict[str, Any] | None, optional
        The keyword arguments to pass to cv2.HoughLinesP.
    do_skeletonize : bool, optional
        Whether to skeletonize the image before applying Hough transform.
    gaussian_size : int, optional
        The size of the Gaussian blur to apply to the image. Default is 5.

    Returns
    -------
    list[list[dict[str, Any]]]
        A list of two lists of clusters:
        - Close-to-horizontal clusters
        - Close-to-vertical clusters
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (gaussian_size, gaussian_size), 0)
    image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, gaussian_size, 2)
    if do_skeletonize:
        image_gray = skeletonize(image_gray)


    # get hough lines
    default_hough_kwargs = {
        "rho": 1,
        "theta": np.pi / 180,
        "threshold": 200,
        "minLineLength": 32,
        "maxLineGap": 20,
    }
    hough_kwargs = {**default_hough_kwargs, **(hough_kwargs or {})}

    lines = cv2.HoughLinesP(image_gray, **hough_kwargs)

    if lines is None:
        return []

    dominant_line_clusters = cluster_lines_by_endpoints([line[0] for line in lines], image.shape, distance_threshold=20.0)
    cluster_lists = select_clusters_by_angle(dominant_line_clusters, angle_threshold=10.0)

    return cluster_lists


def load_and_correct_image(image_path: Path) -> np.ndarray:
    """
    Loads and corrects the perspective of the image.

    Parameters
    ----------
    image_path : Path
        The path to the image to load and correct the perspective of.

    Returns
    -------
    np.ndarray
        The corrected image.
    """
    image = cv2.imread(image_path)
    cluster_lists = extract_potential_grid_line_clusters(image)

    if len(cluster_lists) < 2:
        return image

    cluster_list_h, cluster_list_v = cluster_lists

    # use the line selection + ransac to estimate projective transformation
    lines_h = lines_to_homogeneous([line["mean"] for line in cluster_list_h])
    lines_v = lines_to_homogeneous([line["mean"] for line in cluster_list_v])

    vp1 = ransac_vanishing_point(lines_h)
    vp2 = ransac_vanishing_point(lines_v)

    H = estimate_rectifying_homography(vp1, vp2)
    H_adj, new_size = adjust_homography_to_view(image.shape, H)

    image_corrected = cv2.warpPerspective(image, H_adj, new_size, flags=cv2.INTER_LINEAR)

    return image_corrected


def detect_grid_boxes_with_dominant_lines(image_corrected: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect grid boxes (squares) in the image using dominant lines.

    Parameters
    ----------
    image_corrected : np.ndarray
        The corrected image to detect grid boxes in.

    Returns
    -------
    list[tuple[int, int, int, int]]
        The list of detected grid boxes (x, y, w, h).
    """
    cluster_lists = extract_potential_grid_line_clusters(image_corrected, hough_kwargs={"threshold": 160})

    if len(cluster_lists) < 2:
        return []

    cluster_list_h, cluster_list_v = cluster_lists

    grid_mask = np.zeros_like(image_corrected[:, :, 0])

    for cluster in cluster_list_h:
        x1, y1, x2, y2 = cluster["mean"]
        cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 1)

    for cluster in cluster_list_v:
        x1, y1, x2, y2 = cluster["mean"]
        cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 1)

    grid_mask[:2, :] = 255
    grid_mask[-2:, :] = 255
    grid_mask[:, :2] = 255
    grid_mask[:, -2:] = 255

    grid_mask = cv2.bitwise_not(grid_mask)
    grid_mask = cv2.erode(grid_mask, np.ones((3, 3), np.uint8), iterations=1)

    # find connected components
    stats = cv2.connectedComponentsWithStats(grid_mask)[2]
    potential_squares = []

    for x, y, w, h, _ in stats:
        if w > 192 or h > 192 or w < 32 or h < 32:
            continue
        aspect_ratio = float(w) / h
        if 0.5 <= aspect_ratio <= 2.0:
            potential_squares.append((int(x), int(y), int(w), int(h)))

    # deduplicate if a square is inside another square
    potential_squares.sort(key=lambda x: (x[1], x[0]))

    keep_squares = [True] * len(potential_squares)
    for i, (x1, y1, w1, h1) in enumerate(potential_squares):
        for j, (x2, y2, w2, h2) in enumerate(potential_squares):
            if i == j:
                continue
            if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
                keep_squares[i] = False

    squares = [x for i, x in enumerate(potential_squares) if keep_squares[i]]

    return squares


def detect_grid_boxes_with_thresholding(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect grid boxes (squares) in the image using contour detection.

    Parameters
    ----------
    image : np.ndarray
        The image to detect grid boxes in.

    Returns
    -------
    list[tuple[int, int, int, int]]
        The list of detected grid boxes (x, y, w, h).
    """
    # Create mask where we're only interested in really black pixels
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    mask = np.zeros_like(blue)
    mask[(blue < 10) & (green < 10) & (red < 10)] = 255

    mask[:2, :] = 255
    mask[-2:, :] = 255
    mask[:, :2] = 255
    mask[:, -2:] = 255

    # invert the mask and erode + dilate it
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find connected components
    stats = cv2.connectedComponentsWithStats(mask)[2]
    potential_squares = []

    for x, y, w, h, _ in stats:
        if w > 192 or h > 192 or w < 24 or h < 24:
            continue
        aspect_ratio = float(w) / h
        if 0.5 <= aspect_ratio <= 2.0:
            potential_squares.append((int(x), int(y), int(w), int(h)))

    # deduplicate if a square is inside another square
    potential_squares.sort(key=lambda x: (x[1], x[0]))

    keep_squares = [True] * len(potential_squares)
    for i, (x1, y1, w1, h1) in enumerate(potential_squares):
        for j, (x2, y2, w2, h2) in enumerate(potential_squares):
            if i == j:
                continue
            if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
                keep_squares[i] = False

    squares = [x for i, x in enumerate(potential_squares) if keep_squares[i]]

    return squares


def detect_grid_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect grid boxes (squares) in the image using dominant lines.

    If the thresholding method fails, we fall back to the dominant lines method.

    Parameters
    ----------
    image : np.ndarray
        The image to detect grid boxes in.

    Returns
    -------
    list[tuple[int, int, int, int]]
        The list of detected grid boxes (x, y, w, h).
    """
    initial_boxes = detect_grid_boxes_with_thresholding(image)
    if len(initial_boxes) > 0:
        return initial_boxes
    return detect_grid_boxes_with_dominant_lines(image)


def match_boxes_to_templates(
    image: np.ndarray,
    box_coords_list: list[tuple[int, int, int, int]],
    session: ort.InferenceSession,
    embeddings: np.ndarray,
) -> list[tuple[int, tuple[int, int], tuple[int, int], float]]:
    """
    Match boxes to templates using a pretrained model.

    Parameters
    ----------
    image : np.ndarray
        The image to match boxes to templates in.
    box_coords_list : list[tuple[int, int, int, int]]
        The list of box coordinates to match to templates.
    session : ort.InferenceSession
        The ONNX session to use for embedding boxes.
    embeddings : faiss.IndexFlatL2
        The embeddings to use for nearest neighbor search.

    Returns
    -------
    list[tuple[str, tuple[int, int], tuple[int, int], float]]
        The list of matches (template_index, pt, (w, h), score).
    """
    # Preprocessing transforms
    preprocess_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Process boxes
    matches = []

    with torch.no_grad():
        for box_coords in box_coords_list:
            x, y, w, h = box_coords
            box_region = image[y : y + h, x : x + w]

            # Convert to RGB and preprocess
            box_rgb = cv2.cvtColor(box_region, cv2.COLOR_BGR2RGB)

            box_tensor = preprocess_transforms(box_rgb).unsqueeze(0)

            # Get embedding
            box_embedding = session.run(None, {"input": box_tensor.numpy()})[0]

            # Get nearest neighbor
            distances = np.linalg.norm(embeddings - box_embedding, axis=1)
            best_idx = np.argmin(distances)
            best_score = distances[best_idx]

            # Store matches for this box
            matches.append((best_idx, (x, y), (w, h), float(best_score)))

    return matches


def convert_template_names(
    matches: list[tuple[int, tuple[int, int], tuple[int, int], float]],
    item_info: dict[str, dict[str, Any]],
    item_label_ids_path: Path,
) -> list[tuple[str, tuple[int, int], tuple[int, int], float]]:
    """
    Converts template indices to template names.

    Parameters
    ----------
    matches : list[tuple[int, tuple[int, int], tuple[int, int], float]]
        The list of matches to convert (template_index, pt, (w, h), score).
    item_info : dict[str, dict[str, Any]]
        The item info to use for converting template indices to template names.

    Returns
    -------
    list[tuple[str, tuple[int, int], tuple[int, int], float]]
        The list of matches with template names (template_name, pt, (w, h), score).
    """
    with open(item_label_ids_path, "r") as f:
        template_indices = json.load(f)

    new_matches = []

    for template_index, pt, (w, h), score in matches:
        template_id = template_indices[template_index][0]
        if template_id in item_info:
            info = item_info[template_id]
            new_matches.append((info["Name"], pt, (w, h), score))

    # sort by y, and then x coordinate
    new_matches.sort(key=lambda x: (x[1][1], x[1][0]))

    return new_matches


def draw_matches(
    image: np.ndarray,
    matches: list[tuple[str, tuple[int, int], tuple[int, int], float]],
) -> np.ndarray:
    """
    Draw rectangles around matched regions.

    Parameters
    ----------
    image : np.ndarray
        The image to draw the matches on.
    matches : list[tuple[str, tuple[int, int], tuple[int, int], float]]
        The list of matches to draw (template_name, pt, (w, h), score).

    Returns
    -------
    np.ndarray
        The image with the matches drawn on it.
    """
    output = image.copy()

    for template_name, pt, (w, h), _ in matches:
        # Draw rectangle around match
        cv2.rectangle(output, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

        # Add template name inside rectangle at 45 degree angle
        text_img = np.zeros_like(image)
        text_center = (pt[0] + 5, pt[1] + 5)
        rotation_matrix = cv2.getRotationMatrix2D(text_center, 315, 1.0)
        cv2.putText(
            text_img,
            template_name,
            text_center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1,
        )
        text_img = cv2.warpAffine(
            text_img,
            rotation_matrix,
            (output.shape[1], output.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cv2.add(output, text_img, output)

    return output


def main() -> None:
    """
    Main function to detect items in an image.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image-dir", type=str, default="resources/test_images")
    parser.add_argument(
        "--output-image-dir", type=str, default="resources/output/test_images"
    )
    parser.add_argument(
        "--output-json-dir", type=str, default="resources/output/test_images_json"
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default="item-recognition-web/public/icon_embeddings",
    )
    parser.add_argument(
        "--model-path", type=str, default="item-recognition-web/public/embedder.onnx"
    )
    parser.add_argument(
        "--labels-path", type=str, default="item-recognition-web/src/labels"
    )
    args = parser.parse_args()

    input_image_dir = Path(args.input_image_dir)
    output_image_dir = Path(args.output_image_dir)
    output_json_dir = Path(args.output_json_dir)
    embeddings_path = Path(args.embeddings_path)
    model_path = Path(args.model_path)
    labels_path = Path(args.labels_path)
    item_info_path = labels_path / "item_info_truncated.json"
    item_label_ids_path = labels_path / "item_label_ids.json"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    # load the embeddings and the model
    embeddings = np.fromfile(
        embeddings_path.with_suffix(".qvals.bin"), dtype=np.uint8
    ).reshape(-1, 576)
    starts = np.fromfile(embeddings_path.with_suffix(".starts.bin"), dtype=np.float32)
    steps = np.fromfile(embeddings_path.with_suffix(".steps.bin"), dtype=np.float32)
    embeddings = starts + embeddings * steps

    session = ort.InferenceSession(model_path)

    # load the item info
    with open(item_info_path, "r") as f:
        item_info = json.load(f)

    image_paths = [
        p
        for file_ext in ["png", "webp", "jpg", "jpeg"]
        for p in input_image_dir.glob(f"*.{file_ext}")
        if p.is_file()
    ]

    for image_path in tqdm(
        image_paths, desc="Processing Images", total=len(image_paths)
    ):
        try:
            image = load_and_correct_image(image_path)
            grid_boxes = detect_grid_boxes(image)
            matches = match_boxes_to_templates(image, grid_boxes, session, embeddings)
            conv_matches = convert_template_names(matches, item_info, item_label_ids_path)

            with open(output_json_dir / image_path.with_suffix(".json").name, "w") as f:
                json.dump(conv_matches, f, indent=4)

            matches_img = draw_matches(image, conv_matches)
            cv2.imwrite(output_image_dir / image_path.name, matches_img)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")


if __name__ == "__main__":
    main()
