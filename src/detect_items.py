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


def detect_grid_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
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
        image = cv2.imread(image_path)
        grid_boxes = detect_grid_boxes(image)
        matches = match_boxes_to_templates(image, grid_boxes, session, embeddings)
        conv_matches = convert_template_names(matches, item_info, item_label_ids_path)

        with open(output_json_dir / image_path.with_suffix(".json").name, "w") as f:
            json.dump(conv_matches, f, indent=4)

        matches_img = draw_matches(image, conv_matches)
        cv2.imwrite(output_image_dir / image_path.name, matches_img)


if __name__ == "__main__":
    main()
