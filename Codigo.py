# ================================================================
# iris_flash_full_pipeline_masked_final_pupil_darkneighbors.py
# Author: Pedro Joaquin Sanchez Gutierrez
#
# Description:
# Complete image-processing pipeline for:
# (1) iris localization,
# (2) explicit pupil detection,
# (3) multi-stage flash reflection detection,
# (4) dark-neighbor based diffusion smoothing, and
# (5) refined inpainting using feather-blended texture integration.
#
# This pipeline is designed for cleaning ophthalmic images affected
# by flash reflections while preserving iris/pupil structure.
# ================================================================

import cv2
import numpy as np
import math
from EyeTrackerMain import process_frame


# ===================== CONFIG =====================

IMAGE_PATH = r"C:\Users\HP OMEN\Documents\Maestria\Primer semestre\Datos Ojos\Eyes test\codigo\Universidad de Coimbra\Detector de pupila\Eyes_pipeline\ojo_flash7.png"

GAUSS_KERNEL = 7
DELTA_R = 6
MAX_R = 195
DELTA_LUM = 12
GLOBAL_RATIO_THR = 0.95
SAMPLES = 720
NEIGHBOR_RADIUS = 20
DIFFUSION_ITER = 5
FLASH_THR_RATIO = 0.97


# ===================== UTILS =====================

def crop_to_aspect_ratio(image, width=800, height=700):
    """
    Crops and resizes the input image to a fixed aspect ratio.
    """
    h, w = image.shape[:2]
    desired = width / height
    current = w / h

    if current > desired:
        new_w = int(desired * h)
        offset = (w - new_w) // 2
        cropped = image[:, offset:offset + new_w]
    else:
        new_h = int(w / desired)
        offset = (h - new_h) // 2
        cropped = image[offset:offset + new_h, :]

    return cv2.resize(cropped, (width, height))


def circle_intensities(gray, cx, cy, r, n=SAMPLES):
    """
    Samples grayscale intensities along a circular contour.
    """
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    values = []
    h, w = gray.shape

    for theta in thetas:
        x = int(cx + r * math.cos(theta))
        y = int(cy + r * math.sin(theta))

        if 0 <= x < w and 0 <= y < h:
            values.append(gray[y, x])

    return np.array(values, dtype=np.float32)


# ===================== PUPIL DETECTION =====================

def detect_pupil_from_iris(gray, ellipse):
    """
    Detects the pupil inside the iris region using Otsu thresholding.
    """
    (cx, cy), (a, b), angle = ellipse

    h, w = gray.shape

    x0 = max(0, int(cx - a / 2))
    x1 = min(w, int(cx + a / 2))
    y0 = max(0, int(cy - b / 2))
    y1 = min(h, int(cy + b / 2))

    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return None

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    _, roi_thr = cv2.threshold(
        roi_blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    roi_thr = cv2.morphologyEx(
        roi_thr,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        roi_thr,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    pupil_contour = max(contours, key=cv2.contourArea)

    (px, py), pr = cv2.minEnclosingCircle(pupil_contour)

    return int(px) + x0, int(py) + y0, int(pr)


# ===================== FLASH DETECTION =====================

def detect_reflections_area_full(
    img_bgr,
    cx,
    cy,
    r,
    thickness=6,
    thr_ratio=FLASH_THR_RATIO,
    expand_px=9
):
    """
    Detects bright flash reflections around a circular region.
    """
    max_channel = np.max(img_bgr, axis=2)
    global_max = np.max(max_channel)
    threshold_value = int(global_max * thr_ratio)

    mask = np.zeros_like(max_channel, dtype=np.uint8)

    cv2.circle(
        mask,
        (cx, cy),
        r + thickness // 2,
        255,
        thickness
    )

    bright = cv2.bitwise_and(max_channel, mask)

    _, bright_mask = cv2.threshold(
        bright,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    grad_x = cv2.Sobel(max_channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(max_channel, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = cv2.magnitude(grad_x, grad_y)

    grad_norm = cv2.normalize(
        grad_mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    halo_zone = cv2.dilate(
        bright_mask,
        np.ones((expand_px, expand_px), np.uint8),
        iterations=1
    )

    halo_candidates = cv2.bitwise_and(grad_norm, halo_zone)

    halo_mask = np.zeros_like(bright_mask)
    halo_mask[halo_candidates > 40] = 255

    expanded_mask = cv2.bitwise_or(bright_mask, halo_mask)

    ys, xs = np.where(expanded_mask > 0)

    if len(xs) > 0:
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)

        dist = np.sqrt((xs - mean_x) ** 2 + (ys - mean_y) ** 2)
        r_equiv = int(np.mean(dist))

        expand_size = max(1, int(r_equiv * 0.30))

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (expand_size, expand_size)
        )

        expanded_mask = cv2.dilate(
            expanded_mask,
            kernel,
            iterations=1
        )

    expanded_mask = cv2.morphologyEx(
        expanded_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )

    overlay = img_bgr.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 2] = 255

    blended = cv2.addWeighted(overlay, 1.0, red_layer, 0.4, 0)
    mask3 = cv2.merge([expanded_mask] * 3)

    overlay = np.where(mask3 > 0, blended, overlay)

    points = list(zip(xs, ys))

    return expanded_mask, points, overlay


# ===================== DIFFUSION =====================

def weighted_diffusion_smooth(
    image,
    mask,
    radius=NEIGHBOR_RADIUS,
    iterations=DIFFUSION_ITER
):
    """
    Smooths the flash reflection regions using neighboring darker pixels.
    """
    result = image.copy()

    h, w = mask.shape
    center_y, center_x = h // 2, w // 2
    iris_radius_limit = int(0.47 * min(h, w))

    for _ in range(iterations):
        smoothed = result.copy()

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        padded = cv2.copyMakeBorder(
            result,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_REFLECT
        )

        padded_gray = cv2.copyMakeBorder(
            gray,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_REFLECT
        )

        mask_inv = cv2.bitwise_not(mask)

        padded_mask_inv = cv2.copyMakeBorder(
            mask_inv,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_CONSTANT,
            value=255
        )

        ys, xs = np.where(mask > 0)

        for y, x in zip(ys, xs):
            y0 = y + radius
            x0 = x + radius

            region_color = padded[
                y0 - radius:y0 + radius + 1,
                x0 - radius:x0 + radius + 1
            ]

            region_gray = padded_gray[
                y0 - radius:y0 + radius + 1,
                x0 - radius:x0 + radius + 1
            ]

            region_valid = padded_mask_inv[
                y0 - radius:y0 + radius + 1,
                x0 - radius:x0 + radius + 1
            ]

            valid_idx = region_valid > 0

            if not np.any(valid_idx):
                continue

            yy, xx = np.mgrid[
                -radius:radius + 1,
                -radius:radius + 1
            ]

            dist_to_center = np.sqrt(
                (x + xx - center_x) ** 2 +
                (y + yy - center_y) ** 2
            )

            iris_mask_local = dist_to_center <= iris_radius_limit

            valid_idx = valid_idx & iris_mask_local

            gray_valid = region_gray[valid_idx]

            if gray_valid.size < 5:
                continue

            threshold_dark = np.percentile(gray_valid, 40)

            dark_idx = (region_gray <= threshold_dark) & valid_idx

            if np.count_nonzero(dark_idx) < 5:
                dark_idx = valid_idx

            dark_pixels = region_color[dark_idx]

            if dark_pixels.size > 0:
                smoothed[y, x] = np.mean(dark_pixels, axis=0)

        result = smoothed

    return result


# ===================== INPAINT + FEATHER BLEND =====================

def refine_inpaint_texture(result_img, flash_mask):
    """
    Applies Telea inpainting and blends the corrected region smoothly.
    """
    inpainted = cv2.inpaint(
        result_img,
        flash_mask,
        7,
        cv2.INPAINT_TELEA
    )

    alpha = flash_mask.astype(np.float32) / 255.0

    alpha = cv2.GaussianBlur(
        alpha,
        (41, 41),
        0
    )

    alpha_3c = cv2.merge([alpha] * 3)

    blended = (
        alpha_3c * inpainted.astype(np.float32) +
        (1 - alpha_3c) * result_img.astype(np.float32)
    )

    return blended.astype(np.uint8)


# ===================== MAIN =====================

def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)

    if img is None:
        print("Could not open image.")
        return

    img = crop_to_aspect_ratio(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (GAUSS_KERNEL, GAUSS_KERNEL),
        0
    )

    ellipse = process_frame(img)

    if ellipse is None:
        print("No iris detected.")
        return

    pupil = detect_pupil_from_iris(gray, ellipse)

    if pupil:
        cx, cy, r = pupil
    else:
        (cx, cy), (a, b), _ = ellipse
        r = int(0.25 * (a + b) / 2)

    flash_mask_total = np.zeros(gray.shape, dtype=np.uint8)

    mean_values = []

    for radius in range(0, MAX_R, DELTA_R):
        values = circle_intensities(gray, cx, cy, radius)

        if values.size == 0:
            continue

        flash_mask, _, _ = detect_reflections_area_full(
            img,
            cx,
            cy,
            radius
        )

        flash_mask_total = cv2.bitwise_or(
            flash_mask_total,
            flash_mask
        )

        mean_values.append(np.mean(values))

    smoothed = weighted_diffusion_smooth(
        img,
        flash_mask_total
    )

    corrected = refine_inpaint_texture(
        smoothed,
        flash_mask_total
    )

    cv2.imshow("Original", img)
    cv2.imshow("Corrected", corrected)
    cv2.imshow("Flash mask", flash_mask_total)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()