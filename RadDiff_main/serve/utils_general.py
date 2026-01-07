import hashlib
import os
from typing import Dict, List, Optional, Tuple

import lmdb
from PIL import Image, ImageDraw


def normalize_path_for_hash(path: str) -> str:
    """
    Normalize file paths to the original prefix used in the original experiments.
    This ensures full reproducibility that allows cache key-value retrieval.
    """
    # The canonical prefix
    canonical_prefix = '/pasteur/data/mimic-cxr-jpg-zip/mimic-cxr-jpg-2.1.0.physionet.org/files/'

    # Extract relative path after '/files/'
    if '/files/' in path:
        relative_path = path.split('/files/', 1)[1]
        return canonical_prefix + relative_path

    return path


def resize_image(image: Image.Image, size=(256, 256)) -> Image.Image:
    """
    Resize an image to a specified size while maintaining aspect ratio.
    """
    width, height = image.size
    longer_side = max(width, height)
    
    if longer_side == 0:
        return image
    
    target_size = max(size)  # Use the larger dimension from the size parameter
    scale = target_size / longer_side
    
    # Ensure the longer side is exactly the target size
    if width >= height:
        new_width = target_size
        new_height = round(height * scale)
    else:
        new_height = target_size
        new_width = round(width * scale)
    
    return image.resize((new_width, new_height))


def merge_images_horizontally(images: List[Image.Image], gap: int = 10) -> Image.Image:
    imgs = [resize_image(image) for image in images]
    total_width = sum(img.width for img in imgs) + gap * (len(imgs) - 1)
    height = imgs[0].height

    merged = Image.new("RGB", (total_width, height))

    x_offset = 0
    for img in imgs:
        merged.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    return merged


def merge_images_vertically(images: List[Image.Image], gap: int = 10) -> Image.Image:
    imgs = images
    total_height = sum(img.height for img in imgs) + gap * (len(imgs) - 1)
    width = max(img.width for img in imgs)

    merged = Image.new("RGB", (width, total_height))

    y_offset = 0
    for img in imgs:
        merged.paste(img, (0, y_offset))
        y_offset += img.height + gap

    return merged


def save_data_diff_image(dataset1: List[Dict], dataset2: List[Dict], save_path: str):
    assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
    n_images = len(dataset1)

    # Load images into memory as PIL Image objects
    images_dataset1 = [Image.open(item["path"]) for item in dataset1]
    images_dataset2 = [Image.open(item["path"]) for item in dataset2]

    # Merge images from the same dataset horizontally
    merged_images_dataset1_first = merge_images_horizontally(
        images_dataset1[: n_images // 2]
    )
    merged_images_dataset1_second = merge_images_horizontally(
        images_dataset1[n_images // 2 :]
    )
    merged_images_dataset2_first = merge_images_horizontally(
        images_dataset2[: n_images // 2]
    )
    merged_images_dataset2_second = merge_images_horizontally(
        images_dataset2[n_images // 2 :]
    )

    # Merge the resulting images from different datasets vertically
    final_merged_image = merge_images_vertically(
        [
            merged_images_dataset1_first,
            merged_images_dataset1_second,
            merged_images_dataset2_first,
            merged_images_dataset2_second,
        ]
    )

    # Save the merged image
    final_merged_image.save(save_path)



def resize_images(images, resolution):
    resized_images = []
    for image in images:
        resized_image = image.resize(resolution)
        resized_images.append(resized_image)
    return resized_images



def save_data_diff_image_separate(dataset1: List[Dict], dataset2: List[Dict], save_path1: str, save_path2: str):
    
    assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
    n_images = len(dataset1)

    # remove the file that ends with if '.gstmp' from the inputs
    dataset1 = [item for item in dataset1 if not item["path"].endswith(".gstmp")]
    dataset2 = [item for item in dataset2 if not item["path"].endswith(".gstmp")]
    
    # Load images into memory as PIL Image objects
    images_dataset1 = [Image.open(item["path"]) for item in dataset1]
    images_dataset2 = [Image.open(item["path"]) for item in dataset2]
    
    # Resize original images to 1024x1024
    new_resolution = (1024, 1024)

    images_dataset1 = resize_images(images_dataset1, new_resolution)
    images_dataset2 = resize_images(images_dataset2, new_resolution)

    # Helper function to create a grid of specified dimensions
    def create_grid(images, rows, cols):
        row_images = []
        for i in range(rows):
            row_images.append(merge_images_horizontally(images[i * cols:(i + 1) * cols]))
        return merge_images_vertically(row_images)

    # create a grid of 4x5 for each dataset
    final_merged_image_1 = create_grid(images_dataset1, rows=4, cols=5)
    final_merged_image_2 = create_grid(images_dataset2, rows=4, cols=5)

    final_merged_image_1.save(save_path1)
    final_merged_image_2.save(save_path2)
    
    print("new save path", save_path1)
    print("new save path", save_path2)


def save_data_diff_image_combined(dataset1: List[Dict], dataset2: List[Dict], save_path: str):
    
    assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
    n_images = len(dataset1)

    # remove the file that ends with if '.gstmp' from the inputs
    dataset1 = [item for item in dataset1 if not item["path"].endswith(".gstmp")]
    dataset2 = [item for item in dataset2 if not item["path"].endswith(".gstmp")]
    
    # Load images into memory as PIL Image objects
    images_dataset1 = [Image.open(item["path"]) for item in dataset1]
    images_dataset2 = [Image.open(item["path"]) for item in dataset2]

    new_resolution = (1024, 1024)
    print(f"save_data_diff_image_combined: Resizing to: {new_resolution}")

    images_dataset1 = resize_images(images_dataset1, new_resolution)
    images_dataset2 = resize_images(images_dataset2, new_resolution)

    # Helper function to create a grid of specified dimensions
    def create_grid(images, rows, cols):
        row_images = []
        for i in range(rows):
            row_images.append(merge_images_horizontally(images[i * cols:(i + 1) * cols]))
        return merge_images_vertically(row_images)
    
    # Create separate grids for each dataset
    group1_grid = create_grid(images_dataset1, rows=4, cols=5)
    group2_grid = create_grid(images_dataset2, rows=4, cols=5)
    
    # Combine with larger gap between groups (50px gap instead of default 10px)
    final_merged_image = merge_images_vertically([group1_grid, group2_grid], gap=50)
    
    print(f"save_data_diff_image_combined: Final grid size = {final_merged_image.size}")

    # Save the combined image
    final_merged_image.save(save_path)
    print("Combined image saved to:", save_path)



def crop_and_compose_from_original(
    dataset1: List[Dict],
    dataset2: List[Dict],
    boxes: List[Tuple[float, float, float, float]],
    out_path: str,
    grid_rows: int = 8,
    grid_cols: int = 5,
    draw_boxes: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Crops regions from original images (not grid images) and creates grid layouts.
    Returns a list of grid images, each showing one crop region applied to all original images.
    Optionally also creates grids with red boxes drawn on the original images.

    Args:
        dataset1: List of dictionaries with 'path' keys pointing to original images (first dataset).
        dataset2: List of dictionaries with 'path' keys pointing to original images (second dataset).
        boxes: List of 5 (x1, y1, x2, y2) tuples in normalized [0,1] format.
        out_path: Base path to save the cropped grid images (will append _crop0, _crop1, etc.).
        grid_rows: Number of rows in the output grid layout.
        grid_cols: Number of columns in the output grid layout.
        draw_boxes: If True, also create grids with red boxes drawn on original images.

    Returns:
        Tuple of (cropped_grid_paths, boxed_grid_paths).
        If draw_boxes is False, boxed_grid_paths will be an empty list.
    """
    # Filter out .gstmp files
    dataset1 = [item for item in dataset1 if not item["path"].endswith(".gstmp")]
    dataset2 = [item for item in dataset2 if not item["path"].endswith(".gstmp")]
    
    # Load original images
    original_images_1 = [Image.open(item["path"]) for item in dataset1]
    original_images_2 = [Image.open(item["path"]) for item in dataset2]
    
    # Resize to 1024x1024 to match the save_data_diff_image pipeline
    new_resolution = (1024, 1024)
    print(f"crop_and_compose_from_original: Resizing to: {new_resolution}")
    
    resized_images_1 = resize_images(original_images_1, new_resolution)
    resized_images_2 = resize_images(original_images_2, new_resolution)
    
    # Combine datasets (dataset1 first, then dataset2)
    all_original_images = resized_images_1 + resized_images_2
    
    output_paths = []
    boxed_output_paths = []
    base_name = os.path.splitext(out_path)[0]
    print("base name", base_name)
    extension = '.png'
    
    for crop_idx, (x1, y1, x2, y2) in enumerate(boxes):
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        
        cropped_images = []
        boxed_images = []  # Images with red boxes drawn on them

        for img_idx, original_img in enumerate(all_original_images):
            img_width, img_height = original_img.size
            # print(f"Crop {crop_idx}, Image {img_idx}: Original size = {original_img.size}")
            
            # Convert normalized coordinates to pixel coordinates
            left_px = int(x1 * img_width)
            top_px = int(y1 * img_height)
            right_px = int(x2 * img_width)
            bottom_px = int(y2 * img_height)
            
            # Ensure valid crop bounds
            left_px = max(0, min(left_px, img_width - 1))
            top_px = max(0, min(top_px, img_height - 1))
            right_px = max(left_px + 1, min(right_px, img_width))
            bottom_px = max(top_px + 1, min(bottom_px, img_height))
            
            # Crop the original image
            cropped_original = original_img.crop((left_px, top_px, right_px, bottom_px))
            
            # Resize cropped image using the same logic as resize_image function
            # This maintains aspect ratio and scales to fit within 256x256
            cropped_images.append(cropped_original)

            # Draw red box on the original image if requested
            if draw_boxes:
                # Convert to RGB if not already to ensure proper color rendering
                boxed_img = original_img.copy()
                if boxed_img.mode != 'RGB':
                    boxed_img = boxed_img.convert('RGB')
                draw = ImageDraw.Draw(boxed_img)
                draw.rectangle(
                    [(left_px, top_px), (right_px, bottom_px)],
                    outline=(255, 0, 0),  # Bright red in RGB
                    width=10  # Increased width for better visibility on 1024x1024 images
                )
                boxed_images.append(boxed_img)

        # Create grid layout with cropped images
        if cropped_images:
            row_images = []
            for row in range(grid_rows):
                row_imgs = cropped_images[row * grid_cols:(row + 1) * grid_cols]
                if row_imgs:
                    row_image = merge_images_horizontally(row_imgs)
                    row_images.append(row_image)
            
            if row_images:
                # Split into two groups: first half (dataset1) and second half (dataset2)
                mid_point = grid_rows // 2
                group1_rows = row_images[:mid_point]
                group2_rows = row_images[mid_point:]
                
                # Create separate grids for each group
                group1_grid = merge_images_vertically(group1_rows)
                group2_grid = merge_images_vertically(group2_rows)
                
                # Combine with larger gap between groups (50px gap instead of default 10px)
                final_grid = merge_images_vertically([group1_grid, group2_grid], gap=50)
                print(f"crop_and_compose_from_original: Final grid {crop_idx} size = {final_grid.size}")
                
                # Save the cropped grid
                crop_out_path = f"{base_name}_crop{crop_idx}{extension}"
                os.makedirs(os.path.dirname(crop_out_path), exist_ok=True)
                final_grid.save(crop_out_path)
                output_paths.append(crop_out_path)
                print(f"Cropped grid {crop_idx} saved to: {crop_out_path}")

        # Create grid layout with boxed images
        if draw_boxes and boxed_images:
            row_images_boxed = []
            for row in range(grid_rows):
                row_imgs = boxed_images[row * grid_cols:(row + 1) * grid_cols]
                if row_imgs:
                    row_image = merge_images_horizontally(row_imgs)
                    row_images_boxed.append(row_image)

            if row_images_boxed:
                # Split into two groups: first half (dataset1) and second half (dataset2)
                mid_point = grid_rows // 2
                group1_rows = row_images_boxed[:mid_point]
                group2_rows = row_images_boxed[mid_point:]

                # Create separate grids for each group
                group1_grid = merge_images_vertically(group1_rows)
                group2_grid = merge_images_vertically(group2_rows)

                # Combine with larger gap between groups
                final_grid_boxed = merge_images_vertically([group1_grid, group2_grid], gap=50)
                print(f"crop_and_compose_from_original: Final boxed grid {crop_idx} size = {final_grid_boxed.size}")

                # Save the boxed grid
                boxed_out_path = f"{base_name}_boxed{crop_idx}{extension}"
                os.makedirs(os.path.dirname(boxed_out_path), exist_ok=True)
                final_grid_boxed.save(boxed_out_path)
                boxed_output_paths.append(boxed_out_path)
                print(f"Boxed grid {crop_idx} saved to: {boxed_out_path}")

    return output_paths, boxed_output_paths



def parse_iterative_crop_coords(coord_obj, expected_num: int = 5) -> list:
    """
    Parse coordinate JSON returned by the model and return a single list of (x1, y1, x2, y2) boxes
    normalized to [0,1], to be applied to the combined image (shared for both groups).

    Supports either:
    - {"boxes": [ {x1,y1,x2,y2}, ... ]} with values already in [0,1]
    - {"coordinates": {"group_a": [...], "group_b": [...]}}; prefers group_a, else group_b.

    Ensures exactly `expected_num` boxes (trim or pad with zero boxes), clamps to [0,1], and orders each box so x1<=x2, y1<=y2.
    """

    def to_float(v) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    def coerce_box(box: Dict) -> Tuple[float, float, float, float]:
        x1 = to_float(box.get("x1", box.get("left", 0.0)))
        y1 = to_float(box.get("y1", box.get("top", 0.0)))
        x2 = to_float(box.get("x2", box.get("right", 0.0)))
        y2 = to_float(box.get("y2", box.get("bottom", 0.0)))
        # Clamp to [0,1]
        x1, y1, x2, y2 = clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)
        # Ensure ordering
        x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
        y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
        return (x_lo, y_lo, x_hi, y_hi)

    boxes = []
    if isinstance(coord_obj, dict):
        # Prefer shared boxes
        if isinstance(coord_obj.get("boxes"), list):
            boxes = coord_obj.get("boxes")
        else:
            # Fall back to per-group schema; prefer group_a then group_b
            groups = coord_obj.get("coordinates", coord_obj)
            candidate = None
            for key in ["group_a", "groupA", "A", "GroupA", "Group_A", "a", "A_boxes",
                        "group_b", "groupB", "B", "GroupB", "Group_B", "b", "B_boxes"]:
                if isinstance(groups.get(key), list):
                    candidate = groups.get(key)
                    break
            if isinstance(candidate, list):
                boxes = candidate

    coerced = [coerce_box(b) for b in (boxes or [])]

    # Enforce expected_num by trimming or padding with zeros
    while len(coerced) < expected_num:
        coerced.append((0.0, 0.0, 0.0, 0.0))
    coerced = coerced[:expected_num]

    return coerced


def hash_key(key) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def get_from_cache(key: str, env: lmdb.Environment) -> Optional[str]:
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        value = txn.get(hashed_key.encode())
    if value:
        return value.decode()
    return None


def save_to_cache(key: str, value: str, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        txn.put(hashed_key.encode(), value.encode())
