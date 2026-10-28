from pathlib import Path

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parent


def read_image(path):
    image = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def cylindrical(image):
    """Reduce stretching in wide panoramas using an approximate focal length."""
    h, w = image.shape[:2]
    focal = 0.75 * max(h, w)
    y, x = np.indices((h, w), dtype=np.float32)
    angle = (x - w / 2) / focal
    map_x = focal * np.tan(angle) + w / 2
    map_y = (y - h / 2) / np.cos(angle) + h / 2
    image = cv.remap(image, map_x, map_y, cv.INTER_LINEAR)
    valid = cv.remap(np.full((h, w), 255, np.uint8), map_x, map_y, cv.INTER_LINEAR)
    return image, np.uint8(valid == 255) * 255


def find_transform(image, reference, mask=None, reference_mask=None):
    """Estimate a homography from image to reference, as in lab04_2."""
    orb = cv.ORB_create(nfeatures=6000, fastThreshold=10)
    key1, desc1 = orb.detectAndCompute(cv.cvtColor(image, cv.COLOR_BGR2GRAY), mask)
    key2, desc2 = orb.detectAndCompute(cv.cvtColor(reference, cv.COLOR_BGR2GRAY), reference_mask)
    if desc1 is None or desc2 is None:
        raise ValueError("Not enough features")

    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    pairs = matcher.knnMatch(desc1, desc2, k=2)
    good = [pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
    if len(good) < 10:
        raise ValueError(f"Not enough matches: {len(good)}")

    points1 = np.float32([key1[m.queryIdx].pt for m in good])
    points2 = np.float32([key2[m.trainIdx].pt for m in good])
    matrix, mask = cv.findHomography(points1, points2, cv.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if matrix is None or inliers < 10 or inliers < 0.35 * len(good):
        raise ValueError(f"Unreliable alignment: {inliers}/{len(good)} inliers")
    print(f"  matches: {len(good)}, RANSAC inliers: {inliers}")
    return matrix


def stitch(images, masks):
    # Align adjacent images, then use the middle image as the reference frame.
    transforms = [np.eye(3)]
    for i in range(1, len(images)):
        matrix = find_transform(images[i], images[i - 1], masks[i], masks[i - 1])
        transforms.append(transforms[-1] @ matrix)
    center_inverse = np.linalg.inv(transforms[len(images) // 2])
    transforms = [center_inverse @ matrix for matrix in transforms]

    corners = []
    for image, matrix in zip(images, transforms):
        h, w = image.shape[:2]
        rectangle = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        corners.append(cv.perspectiveTransform(rectangle[None], matrix)[0])
    corners = np.concatenate(corners)
    if not np.isfinite(corners).all():
        raise ValueError("Invalid panorama bounds")
    left, top = np.floor(corners.min(axis=0)).astype(int)
    right, bottom = np.ceil(corners.max(axis=0)).astype(int)
    width, height = int(right - left), int(bottom - top)
    if width <= 0 or height <= 0 or width * height > 25_000_000:
        raise ValueError(f"Invalid panorama size: {width}x{height}")
    translation = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=float)

    total = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for image, mask, matrix in zip(images, masks, transforms):
        h, w = image.shape[:2]
        # Feather the overlap: pixels near a frame's edge get less weight.
        padded = cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
        weight = cv.distanceTransform(padded, cv.DIST_L2, 3)[1:-1, 1:-1]
        weight = np.minimum(weight, 50)
        matrix = translation @ matrix
        warped = cv.warpPerspective(image.astype(np.float32) * weight[..., None],
                                    matrix, (width, height))
        warped_weight = cv.warpPerspective(weight, matrix, (width, height))
        total += warped
        weights += warped_weight

    result = total / np.maximum(weights[..., None], 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    cv.setRNGSeed(0)
    cv.setNumThreads(1)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    for folder in sorted((ROOT / "hw4").glob("building*")):
        files = sorted(folder.glob("*.jpg"))
        if len(files) < 2:
            raise ValueError(f"Need at least two images in {folder}")
        print(f"{folder.name}: {len(files)} images")
        images = [read_image(path) for path in files]
        masks = [np.full(image.shape[:2], 255, np.uint8) for image in images]
        if folder.name in ("building2", "building3"):
            images, masks = map(list, zip(*(cylindrical(image) for image in images)))
        panorama = stitch(images, masks)
        destination = output / f"{folder.name}.jpg"
        ok, data = cv.imencode(".jpg", panorama, [cv.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError(f"Cannot encode {destination}")
        data.tofile(destination)
        print(f"  saved {destination.name}: {panorama.shape[1]}x{panorama.shape[0]}")


if __name__ == "__main__":
    main()
