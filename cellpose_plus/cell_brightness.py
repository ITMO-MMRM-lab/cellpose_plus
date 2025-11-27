import numpy as np
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
import pathlib
import colorsys


def calculate_brightness_values(image, mask, channel, subtract_background=True,
                               rgb_ref_color=None, ref_color_threshold=None,
                               logger=None):
    """
    Calculates brightness values for each cell in the mask.
    
    :param image: image numpy array
    :param mask: mask numpy array
    :param channel: how to calculate brightness (0-red, 1-green, 2-blue, 3-average)
    :param subtract_background: if True, subtracts background brightness
    :param ref_color: tuple (R, G, B) - reference color to exclude
    :param ref_color_threshold: percentage threshold for color matching (0-100)
    :return: tuple (data DataFrame, center_coords list, brightness_map array)
    """
    # Extract brightness channel
    if 0 <= channel <= 2:
        brightness = image[:, :, channel]
    elif channel == 3:
        brightness = np.mean(image, axis=2)
    else:
        raise ValueError("Channel must be 0 (red), 1 (green), 2 (blue), or 3 (gray)")

    # Get unique cell IDs
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids != 0]
    if len(object_ids) == 0:
        return None, None, None
    object_ids_to_process = np.array(object_ids)    

    # Filter cells by reference color if provided
    colors = []
    filtered_object_ids = []

    if rgb_ref_color is not None:
        hsv_ref_color = rgb_to_hsv(rgb_ref_color)
        if logger:
            logger.info(f"Reference color HSV:{hsv_ref_color}")
        for obj_id in object_ids:
            obj_mask = (mask == obj_id)
            # Calculate mean RGB for this cell
            # average_r = int(image[:, :, 0][obj_mask].mean())
            # average_g = int(image[:, :, 1][obj_mask].mean())
            # average_b = int(image[:, :, 2][obj_mask].mean())
            # if logger:
            #     logger.info(f"Mean color RGB{(mean_r, mean_g, mean_b)} of ID:{obj_id} cell")
            average_r = int(np.median(image[:, :, 0][obj_mask]))
            average_g = int(np.median(image[:, :, 1][obj_mask]))
            average_b = int(np.median(image[:, :, 2][obj_mask]))
            if logger:
                logger.info(f"Median color RGB:{(average_r, average_g, average_b)} of cell ID:{obj_id}")
            cell_color = (average_r, average_g, average_b)
            average_h, average_s, average_v = rgb_to_hsv(cell_color)
            if logger:
                logger.info(f"Average color HSV:{(average_h, average_s, average_v)} of cell ID:{obj_id}")
            hsv_cell_color = (average_h, average_s, average_v)

            # Check if cell color is within ref_color_threshold threshold for RGB
            rgb_excluded = True
            if not is_rgb_color_match(cell_color, rgb_ref_color, ref_color_threshold):
                # filtered_object_ids.append(obj_id)
                rgb_excluded = False

            # Check if cell color is within ref_color_threshold threshold for HSV
            hsv_excluded = True
            if not is_hsv_color_match(hsv_cell_color, hsv_ref_color, ref_color_threshold):
                filtered_object_ids.append(obj_id)
                hsv_excluded = False

            colors.append({
                'id': obj_id, 
                'R': average_r, 
                'G': average_g, 
                'B': average_b, 
                'rgbExcluded': rgb_excluded,
                'H': average_h, 
                'S': average_s, 
                'V': average_v, 
                'hsvExcluded': hsv_excluded,
            })

        if len(filtered_object_ids) == 0:
            return None, None, None
        object_ids_to_process = np.array(filtered_object_ids)
    
    # Calculate background brightness
    background_brightness = 0
    if subtract_background:
        filtered = False
        if filtered:
            # Put excluded cells in to the Background
            background_mask = np.ones_like(mask, dtype=bool)
            for obj_id in object_ids_to_process:
                background_mask &= (mask != obj_id)
        else:
            background_mask = (mask == 0)
        
        if np.any(background_mask):
            background_brightness = brightness[background_mask].mean()
    
    # Calculate brightness for each cell
    data = []
    center_coords = []
    
    for obj_id in object_ids:
        obj_mask = (mask == obj_id)
        mean_brightness = brightness[obj_mask].mean() - background_brightness
        intensity = classify_intensity(mean_brightness)
        
        data.append({
            'id': obj_id, 
            'mean_brightness': mean_brightness, 
            'intensity': intensity,
            'excluded': not obj_id in object_ids_to_process,
        })
        
        # Calculate center coordinates
        coords = np.column_stack(np.where(obj_mask))
        center_x = coords[:, 1].mean()
        center_y = coords[:, 0].mean()
        center_coords.append((center_x, center_y))
    
    # Create brightness map
    brightness_map = np.zeros_like(brightness)
    for obj_id in object_ids_to_process:
        obj_mask = (mask == obj_id)
        mean_brightness = next(item['mean_brightness'] for item in data if item['id'] == obj_id)
        # mean_brightness = data.loc[data['id'] == obj_id, 'mean_brightness'].values[0]
        brightness_map[obj_mask] = mean_brightness
    
    return pd.DataFrame(data), center_coords, brightness_map, pd.DataFrame(colors)


def rgb_to_hsv(color):
    r, g, b = [x/255.0 for x in color]  # Normalize to 0-1 range
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (round(h*360), round(s*100), round(v*100))


def is_rgb_color_match(color, ref_color, ref_color_threshold):
    """
    Checks if two RGB colors match within ref_color_threshold threshold.
    
    :param color: tuple (R, G, B) - color to check
    :param ref_color: tuple (R, G, B) - reference color  
    :param ref_color_threshold: percentage threshold (0-100)
    :return: True if colors match within threshold
    """
    # Calculate percentage difference for each channel
    for c1, c2 in zip(color, ref_color):
        # Handle case when reference value is 0
        if c2 == 0:
            if c1 > ref_color_threshold * 2.55:  # Convert percentage to 0-255 range
                return False
        else:
            percent_diff = abs((c1 - c2) / c2) * 100
            if percent_diff > ref_color_threshold:
                return False
    return True


def is_hsv_color_match(color, ref_color, ref_color_threshold):
    """
    Checks if two HSV colors match within ref_color_threshold threshold.
    
    :param color: tuple (H, S, V) - color to check
    :param ref_color: tuple (H, S, V) - reference color  
    :param ref_color_threshold: percentage threshold (0-100)
    :return: True if colors match within threshold
    """
    h1, s1, v1 = color
    h2, s2, v2 = ref_color
    
    # Handle Hue (0-360 degrees, circular)
    # Calculate minimum angular distance between hues
    h_diff = abs(h1 - h2)
    if h_diff > 180:
        h_diff = 360 - h_diff
    
    # Convert angular difference to percentage (180° = 100%)
    h_percent_diff = (h_diff / 180) * 100
    if h_percent_diff > ref_color_threshold:
        return False
    
    # # Handle Saturation (0-100)
    # if s2 == 0:
    #     if s1 > ref_color_threshold:
    #         return False
    # else:
    #     s_percent_diff = abs((s1 - s2) / s2) * 100
    #     if s_percent_diff > ref_color_threshold:
    #         return False
    
    # # Handle Value/Brightness (0-100)
    # if v2 == 0:
    #     if v1 > ref_color_threshold:
    #         return False
    # else:
    #     v_percent_diff = abs((v1 - v2) / v2) * 100
    #     if v_percent_diff > ref_color_threshold:
    #         return False
    
    return True


def calculate_cell_brightness(image, mask, filepath, channel, 
                              subtract_background=True,
                              ref_color=None, ref_color_threshold=None, 
                              logger=None):
    """
    Calculates cell brightness and saves results to `*image-folder*/*image-name*/brightness/`

    :param image: image numpy array
    :param mask: current mask numpy array
    :param filepath: path to source image file
    :param channel: how to calculate brightness \n
        - `0` - red channel only
        - `1` - green channel only
        - `2` - blue channel only
        - `3` - average brightness of all channels
    :param subtract_background: if True, subtracts background brightness from each cell
    :param ref_color: tuple (R, G, B) - reference color for background cells (0-255 range)
    :param ref_color_threshold: percentage threshold for color matching (0-100)
     """
    
    # # ref_color = (205, 192, 204)
    # ref_color = (213, 203, 223)
    # ref_color_threshold=10

    if not image.any():
        if logger:
            logger.info("No image data provided")
        return "No image data provided"
  
    image = image[0]
    mask = mask[0]
    
    # Log reference color filtering if enabled
    if ref_color is not None and logger:
        logger.info(f"Filtering cells with color close to RGB{ref_color} ± {ref_color_threshold}%")
    
    # Calculate brightness values
    try:
        data, center_coords, brightness_map, colors = calculate_brightness_values(
            image, mask, channel, subtract_background, 
            ref_color, ref_color_threshold, 
            logger
        )
    except ValueError as e:
        if logger:
            logger.info(f"Invalid channel specified: {e}")
        raise
    
    if data is None:
        if logger:
            logger.info("No cells found in the mask")
        return "No cells found in the mask"
    
    # Log filtering results
    if ref_color is not None and logger:
        total_cells = len(np.unique(mask[mask != 0]))
        remaining_cells = len(data)
        filtered_out = total_cells - remaining_cells
        logger.info(f"Filtered out {filtered_out} cells as background (total: {total_cells}, remaining: {remaining_cells})")
    
    # Prepare filenames
    channel_filename = ["red", "green", "blue", "gray"]
    filename = f"cell_brightness_{channel_filename[channel]}"
    
    # Create directories
    results_dir = os.path.splitext(filepath)[0]
    brightness_dir = os.path.join(results_dir, "brightness")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(brightness_dir):
        os.makedirs(brightness_dir)
    
    # Save CSV with additional color info
    data.to_csv(os.path.join(brightness_dir, f"{filename}.csv"), index=False)
    
    # Save filtering parameters if reference color was used
    if ref_color is not None:
        colors.to_csv(os.path.join(brightness_dir, "cell_colors.csv"), index=False)
    
    if logger:
        logger.info("Brightness of cells is calculated")
    
    # Create and save colormap (only for remaining cells)
    object_ids = data['id'].values
    colormap_mask = Image.fromarray(create_colormap_mask(mask))
    im_masks = label_image(colormap_mask, object_ids, center_coords)
    im_masks.save(os.path.join(brightness_dir, "mask_colormap.png"))

    if ref_color:
        colormap_mask = Image.fromarray(create_real_colormap_mask(mask, colors))
        im_masks = label_image(colormap_mask, object_ids, center_coords)
        im_masks.save(os.path.join(brightness_dir, "mask_real_colormap.png"))

        filtered_object_ids = data[data['excluded'] == False]['id'].values
        filtered_mask = np.zeros_like(mask)
        for obj_id in filtered_object_ids:
            filtered_mask[mask == obj_id] = obj_id
        colormap_mask = Image.fromarray(create_real_colormap_mask(filtered_mask, colors))
        im_masks = label_image(colormap_mask, filtered_object_ids, center_coords)
        im_masks.save(os.path.join(brightness_dir, "filtered_mask_colormap.png"))

    if logger:
        logger.info("Colormap is created")
    
    # Create and save brightness visualization map
    brightness_image = create_brightness_visualization(brightness_map, data, center_coords)
    brightness_image.save(os.path.join(brightness_dir, f"{filename}_visualization_map.png"))
    
    if logger:
        logger.info("Brightness visualization map is created")
    
    h_score, strong_ratio, mean_ratio, weak_ratio, result = calculate_h_score_values(data)
    result.to_csv(os.path.join(brightness_dir, f"h_score.csv"), index=False)

    if logger:
        logger.info("H-score is calculated")

    return "Done! Results saved to folder"


def create_brightness_visualization(brightness_map, data, center_coords):
    """
    Creates a brightness visualization image with labeled values.
    
    :param brightness_map: numpy array with brightness values
    :param data: DataFrame with brightness data
    :param center_coords: list of center coordinates for labels
    :return: PIL Image object
    """
    # Make brightness map more contrastive
    non_zero_brightness = brightness_map[brightness_map > 0]
    if len(non_zero_brightness) == 0:
        return Image.fromarray(np.zeros_like(brightness_map, dtype=np.uint8))
    
    vmin = non_zero_brightness.min()
    # vmin = brightness_map[brightness_map > 0].min()
    vmax = brightness_map.max()
    brightness_norm = (brightness_map - vmin) / (vmax - vmin + 1e-8)
    brightness_norm = np.clip(brightness_norm, 0, 1)
    
    gamma = 0.5
    brightness_gamma = np.power(brightness_norm, gamma)
    brightness_map_scaled = (brightness_gamma * 255).astype(np.uint8)
    
    # Create image and add labels
    brightness_image = Image.fromarray(brightness_map_scaled)
    brightness_image = brightness_image.convert("L")
    brightness_labels = data['mean_brightness'].map(lambda x: f"{x:.2f}")
    brightness_image = label_image(brightness_image, brightness_labels, center_coords, color=255)
    
    return brightness_image


def create_colormap_mask(mask):
    colormap = ((np.random.rand(1000000,3)*0.8+0.1)*255).astype(np.uint8)
    tmp_mask = np.copy(mask).astype(np.uint8)

    colors = colormap[:tmp_mask.max(), :3]
    cellcolors = np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8)

    layerz = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)

    new_tmp_mask = tmp_mask[np.newaxis,...]

    layerz[...,:3] = cellcolors[new_tmp_mask[0],:]
    layerz[...,3] = 128 * (new_tmp_mask[0]>0).astype(np.uint8)

    return layerz


def create_real_colormap_mask(mask, colors):
    """
    Creates a colormap mask using real colors from the colors DataFrame.
    
    :param mask: mask numpy array
    :param colors: DataFrame with columns 'id', 'R', 'G', 'B'
    :return: colored mask array
    """
    # Create output array with RGBA channels
    layerz = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)
    
    # Set background to white
    layerz[..., :3] = 255
    layerz[..., 3] = 0  # Transparent background
    
    # Apply colors for each cell
    for _, row in colors.iterrows():
        obj_id = int(row['id'])
        r = int(row['R'])
        g = int(row['G'])
        b = int(row['B'])
        
        # Find pixels belonging to this cell
        cell_mask = (mask == obj_id)
        
        # Apply color to these pixels
        layerz[cell_mask, 0] = r
        layerz[cell_mask, 1] = g
        layerz[cell_mask, 2] = b
        layerz[cell_mask, 3] = 255  # Opaque for cells
    
    return layerz

def label_image(image, values, coords, color=(255, 255, 255)):
    image_labeled = image.copy()

    font_path = pathlib.Path.home().joinpath(".cellpose_plus", "DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), size=10)
    
    I1 = ImageDraw.Draw(image_labeled)
    
    for value, coord in zip(values, coords):
        I1.text((coord[0], coord[1]), str(value), 
                anchor="mb",
                fill=color,
                font=font)

    return image_labeled


def classify_intensity(brightness):
    BRIGHTNESS_RANGES = (
        ("strong", (34, 115)),
        ("mean", (115, 147)),
        ("weak", (147, 191))
    )

    for level, (low, high) in BRIGHTNESS_RANGES:
        if low <= brightness <= high:
            return level
    return "out_of_range"


def calculate_h_score_values(data):
    """
    Calculates h_score and ratios.
    
    :param data: DataFrame
    :return: tuple (h_score, strong_ratio, mean_ratio, weak_ratio, result DataFrame)
    """
    # Count for intensity
    counts_intensity = data['intensity'].value_counts()
    total_count = len(data)

    # Counting ratios
    strong_count = counts_intensity.get('strong', 0)
    mean_count = counts_intensity.get('mean', 0)
    weak_count = counts_intensity.get('weak', 0)

    strong_ratio = strong_count / total_count * 100 if total_count > 0 else 0
    mean_ratio = mean_count / total_count * 100 if total_count > 0 else 0
    weak_ratio = weak_count / total_count * 100 if total_count > 0 else 0
        
    # Counting H-score
    h_score = 1 * weak_ratio + 2 * mean_ratio + 3 * strong_ratio

    result = {
        'metric': ['h_score', 'weak', 'mean', 'strong'],
        'value': [h_score, weak_ratio, mean_ratio, strong_ratio],
        'count': [total_count, weak_count, mean_count, strong_count]
    }

    return h_score, strong_ratio, mean_ratio, weak_ratio, pd.DataFrame(result)
