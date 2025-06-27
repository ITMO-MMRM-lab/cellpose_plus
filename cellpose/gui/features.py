import sys, os, pathlib, warnings, datetime, time, copy, math
from qtpy import QtGui
from qtpy.QtWidgets import QAction, QMenu

from . import symmetry
from ..utils import download_font
import pandas as pd
import numpy as np
from scipy.stats import mode
import cv2

from scipy.ndimage import find_objects
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, convex_hull_plot_2d
from scipy import ndimage
import diplib as dip
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except:
    MATPLOTLIB = False


class FeatureExtraction:
    """
    A class for extracting and processing features from image masks in a computational context.

    This class provides various methods to analyze, manipulate, and save image masks, including
    functionalities for scaling contours, generating color maps, labeling images, and calculating
    metrics for regions in binary masks. It is designed to facilitate the examination of cellular
    structures in images and supports integration with a graphical user interface (GUI).

    Attributes:
        masks_menu: Pointer to the masks menu in the GUI.
        images_menu: Pointer to the images menu in the GUI.
        cyto_mask_index: Index for cytoplasm masks.
        nuclei_mask_index: Index for nucleus masks.
        current_image_mask: Currently selected image mask.
        current_model: Current model being used for processing.
        temp_masks: Temporary list for masks.

    Methods:
        __init__
        save_temp_output
        scale_contour
        save_labeled_masks
        create_colormap_mask
        image_labeling
        mask_indexing
        create_colormap
        find_overlap
        matched_indices
        get_metrics
        out_concat
        get_voronoi_entropy
        save_metrics
        calculate_metrics
        select_mask
        select_image

    This class methods allow for saving outputs, calculating overlaps and matching indices for
    masks, generating various metrics and visual representations, and managing GUI interactions
    for mask selections and image displays.
    """

    def __init__(self):
        """
        Initializes the FeatureExtraction class.

            This constructor sets up the necessary attributes for the FeatureExtraction class, including pointers to the masks and images menus, indices for cyto and nucleus masks, the current image mask, and the current model. It also initializes a temporary list for masks and calls a function to download the font if the GUI is not running.

            Parameters:
                None

            Returns:
                None
        """
        super(FeatureExtraction, self).__init__()

        self.main_masks_menu = None  # Pointer to masks menu
        self.main_images_menu = None  # Pointer to images menu

        self.indexCytoMask = -1
        self.indexNucleusMask = -1
        self.currentImageMask = ""
        self.current_model = ""
        self.temp_masks = []

        download_font()  # If running without GUI

    def save_temp_output(self, masks="", image="", model_name="", gui_self=""):
        """
        Save temporary output masks and images.

            This method handles the temporary storage of output masks and images based on the provided parameters.
            It creates submenu actions for selecting masks and logs the storing of temporary masks.

            Args:
                masks: A string representing the existing masks, defaulting to an empty string.
                image: A string representing the image to be processed, defaulting to an empty string.
                model_name: A string representing the model name. If not provided, it defaults to an empty string.
                gui_self: An instance of the GUI context used to interact with the user interface elements.

            Returns:
                None: This method does not return a value but performs actions to store masks and update the GUI.
        """
        d = datetime.datetime.now()
        temp_output_name = gui_self.current_model if model_name == "" else model_name

        if image == "":
            mask_names = [
                mask_name[0]
                for mask_name in self.temp_masks
                if temp_output_name in mask_name[0]
                and mask_name[0][len(temp_output_name)] == "_"
            ]
            new_mask_names = temp_output_name + "_" + str(len(mask_names) + 1)
            subMenu = self.main_masks_menu.addMenu("&" + new_mask_names)

            cytoAction = QAction("Select as main mask (cytoplasm)", gui_self)
            cytoAction.triggered.connect(
                lambda checked, subMenu=subMenu, curr_index=len(
                    self.temp_masks
                ): self.select_mask(subMenu, "primary", curr_index, gui_self)
            )

            nucleiAction = QAction("Select as secondary mask (nucleus)", gui_self)
            nucleiAction.triggered.connect(
                lambda checked, subMenu=subMenu, curr_index=len(
                    self.temp_masks
                ): self.select_mask(subMenu, "secondary", curr_index, gui_self)
            )

            subMenu.addAction(cytoAction)
            subMenu.addAction(nucleiAction)

            self.temp_masks.append((new_mask_names, gui_self.cellpix[-1]))  # masks[-1]
        else:  # elif masks == "":
            if self.indexCytoMask > -1:
                full_name = (
                    temp_output_name + " " + self.temp_masks[self.indexCytoMask][0]
                )
                newImage = QAction(full_name, gui_self)
                newImage.triggered.connect(
                    lambda checked, image=image, name=full_name: self.select_image(
                        gui_self, image, name
                    )
                )
                self.main_images_menu.addAction(newImage)

        gui_self.logger.info(str(temp_output_name) + " mask stored temporarily")

    def scale_contour(self, cnt, scale):
        """
        Scales a contour around its centroid by a specified factor.

            This method takes a contour represented as a set of points and scales
            it relative to its centroid by a given scale factor. The scaling is
            performed in such a way that the centroid remains fixed.

            Args:
                cnt: A numpy array representing the contour points.
                scale: A float value representing the scaling factor.

            Returns:
                A numpy array containing the scaled contour points.
        """
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)

        return cnt_scaled

    def save_labeled_masks(self, gui_self):
        """save masks to *_mask.jpg"""

        # Create results dir
        results_dir = os.path.splitext(gui_self.filename)[0]
        labels_dir = results_dir + "/labels"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)

        slices = find_objects(gui_self.cellpix[0].astype(int))
        for idx in range(gui_self.cellpix[0].max()):
            tmp_cellpix = np.copy(gui_self.cellpix[0])
            tmp_cellpix[idx + 1 != gui_self.cellpix[0]] = 0
            tmp_cellpix[idx + 1 == gui_self.cellpix[0]] = 255

            mask = tmp_cellpix.astype(np.uint8)

            im = Image.fromarray(mask)
            label_name = labels_dir + "/" + str(idx + 1) + ".png"
            im.save(label_name)

        tmp_cellpix = np.copy(gui_self.cellpix[0])
        new_cellpix = np.zeros_like(tmp_cellpix)
        for idx in range(gui_self.cellpix[0].max()):
            tmp_mask = np.copy(gui_self.cellpix[0])
            tmp_mask[idx + 1 != gui_self.cellpix[0]] = 0
            tmp_mask[idx + 1 == gui_self.cellpix[0]] = 255

            contours, _ = cv2.findContours(
                tmp_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [self.scale_contour(contour, 0.95) for contour in contours]
            new_contours = cv2.drawContours(
                new_cellpix, contours, -1, (255, 255, 255), -1
            )
            # new_cellpix[self.scale_contour(contours[0], 0.9) == 255] = 255
            # new_cellpix[tmp_mask == 255] = 255

        mask_tmp = new_cellpix.astype(np.uint8)

        im_tmp = Image.fromarray(mask_tmp)
        label_name = labels_dir + "/" + "total_mask.png"
        im_tmp.save(label_name)

    def create_colormap_mask(self, mask):
        """
        Creates a colormap mask from the given input mask.

            This method generates a random colormap and applies it to the
            provided mask, creating an RGBA image where each color is
            mapped based on the mask's values. The output image has an
            additional alpha channel to indicate the mask’s presence.

            Args:
                mask: The input mask array from which colors will be derived.

            Returns:
                A 4-dimensional array representing the colormap mask with RGBA channels.
        """
        colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(np.uint8)
        tmp_mask = np.copy(mask).astype(np.uint8)

        colors = colormap[: tmp_mask.max(), :3]
        cellcolors = np.concatenate(
            (np.array([[255, 255, 255]]), colors), axis=0
        ).astype(np.uint8)

        layerz = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)

        new_tmp_mask = tmp_mask[np.newaxis, ...]

        layerz[..., :3] = cellcolors[new_tmp_mask[0], :]
        layerz[..., 3] = 128 * (new_tmp_mask[0] > 0).astype(np.uint8)

        return layerz

    def image_labeling(self, im_mask="", im_labels="", coords=""):
        """
        Label an image mask with specified labels and coordinates.

            This method takes an image mask and overlays labels at specified coordinates.
            If no labels are provided, it will use sequential numeric labels. The labeled
            image mask is returned after drawing the specified labels.

            Args:
                im_mask: The image mask to be labeled.
                im_labels: Optional labels to use for each coordinate. If not provided,
                            sequential numeric labels will be used.
                coords: A list of tuples representing the coordinates where the labels
                        should be drawn on the image mask.

            Returns:
                The labeled image mask with the applied labels at the specified coordinates.
        """
        im_mask_labeled = im_mask.copy()

        font_path = pathlib.Path.home().joinpath(".cellpose", "DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), size=20)

        I1 = ImageDraw.Draw(im_mask_labeled)

        for idx in range(0, len(coords)):
            if im_labels == "":
                label_value = str(idx + 1)
            else:
                label_value = str(im_labels[idx])

            I1.text(
                (coords[idx][0], coords[idx][1]),
                label_value,
                anchor="mb",
                fill=(255, 255, 255),
                font=font,
            )

        return im_mask_labeled

    def mask_indexing(self, im_mask, coords):
        """
        Annotates an image mask with indexed coordinates.

            This method adds numbered labels to the specified coordinates on a copy
            of the provided image mask. It uses a TrueType font to draw the labels.

            Args:
                im_mask: The image mask to be annotated.
                coords: A list of coordinate tuples where each label should be placed.

            Returns:
                An image mask with the indexed labels applied.
        """
        im_mask_labeled = im_mask.copy()

        font_path = pathlib.Path.home().joinpath(".cellpose", "DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), size=20)

        I1 = ImageDraw.Draw(im_mask_labeled)

        for idx in range(0, len(coords)):
            I1.text(
                (coords[idx][0], coords[idx][1]),
                str(idx + 1),
                anchor="mb",
                fill=(255, 255, 255),
                font=font,
            )

        return im_mask_labeled

    def create_colormap(mask_cyto, mask_nuclei):
        """
        Creates color maps for cellular images and identifies overlaps.

            This method generates color maps for cytoplasm and nuclei based on input masks,
            and also creates an overlap color map to visualize areas where both masks intersect.

            Args:
                mask_cyto: The mask representing the cytoplasm areas.
                mask_nuclei: The mask representing the nuclei areas.

            Returns:
                A tuple containing three image objects:
                    - im_cyto: An image with the cytoplasm color map.
                    - im_nuclei: An image with the nuclei color map.
                    - im_overlap: An image showing the overlap between the cytoplasm and nuclei color maps.
        """
        # Cyto colormap
        layerz_cyto = create_colormap_mask(mask_cyto)
        im_cyto = Image.fromarray(layerz_cyto)

        # Nuclei colormap
        layerz_nuclei = create_colormap_mask(mask_nuclei)
        im_nuclei = Image.fromarray(layerz_nuclei)

        # Overlap colormap
        layerz_overlap = np.copy(layerz_cyto).astype(np.uint8)
        for idxi in range(0, layerz_overlap.shape[0]):
            for idxj in range(0, layerz_overlap.shape[1]):
                if (layerz_cyto[idxi][idxj] != [255, 255, 255, 0]).all() and (
                    layerz_nuclei[idxi][idxj] != [255, 255, 255, 0]
                ).all():
                    layerz_overlap[idxi][idxj] = [255, 0, 0, 128]
        im_overlap = Image.fromarray(layerz_overlap)

        return im_cyto, im_nuclei, im_overlap

    def find_overlap(self, cyto_mask, nuclei_mask, cyto_nuclei_indices):
        """
        Counts the number of overlapping elements between cytoplasmic and nuclear masks.

            This method iterates through the provided cytoplasmic and nuclear masks and counts
            the number of pixels where the values match the specified indices for cytoplasm and
            nuclei. It essentially identifies how many pixels belong to both the cytoplasm and
            nuclei as defined by the input indices.

            Args:
                cyto_mask: A 2D array representing the cytoplasm mask.
                nuclei_mask: A 2D array representing the nuclei mask.
                cyto_nuclei_indices: A list containing two indices; the first for the cytoplasm
                                     and the second for the nuclei.

            Returns:
                An integer representing the count of overlapping pixels between the cytoplasm
                and nuclei based on the specified indices.
        """
        count = 0
        for idxi in range(0, cyto_mask.shape[0]):
            for idxj in range(0, cyto_mask.shape[1]):
                if (
                    cyto_mask[idxi][idxj] == cyto_nuclei_indices[0]
                    and nuclei_mask[idxi][idxj] == cyto_nuclei_indices[1]
                ):
                    count += 1
        return count

    def matched_indices(
        self, cyto_mask, nuclei_mask, cyto_size, nuclei_size, main_coords
    ):
        """
        Identify matched indices between cytoplasmic and nuclei masks.

            This method compares cytoplasmic and nuclei masks to identify unique matches
            based on their indices. It processes the masks to remove duplicates and ensures
            that each cytoplasmic index corresponds to a single nuclei index. It also
            calculates the ratio of cytoplasmic to nuclei sizes.

            Args:
                cyto_mask: The mask representing the cytoplasm.
                nuclei_mask: The mask representing the nuclei.
                cyto_size: The sizes of the cytoplasmic indices.
                nuclei_size: The sizes of the nuclei indices.
                main_coords: The coordinates associated with the main structure.

            Returns:
                A tuple containing:
                    - A list of cytoplasmic to nuclei size ratios.
                    - A list of coordinates corresponding to the matched indices.
                    - A list of unique matched indices between the cytoplasm and nuclei.
        """
        tmp_cyto = np.copy(cyto_mask)  # .astype(np.uint8)
        tmp_nuclei = np.copy(nuclei_mask)  # .astype(np.uint8)
        tmp_coords = np.copy(main_coords)

        indices_cyto_nuclei = set()

        # Remove duplicates
        for idxi in range(0, tmp_cyto.shape[0]):
            for idxj in range(0, tmp_cyto.shape[1]):
                if tmp_cyto[idxi][idxj] != 0 and tmp_nuclei[idxi][idxj] != 0:
                    indices_cyto_nuclei.add(
                        (tmp_cyto[idxi][idxj], tmp_nuclei[idxi][idxj])
                    )

        indices_cyto_nuclei = list(indices_cyto_nuclei)
        indices_cyto_nuclei.sort()

        # Assure 1 cyto for 1 nuclei
        n = len(indices_cyto_nuclei)
        cnt = 0

        while cnt < n - 1:
            if indices_cyto_nuclei[cnt][0] == indices_cyto_nuclei[cnt + 1][0]:
                to_del = (
                    self.find_overlap(tmp_cyto, tmp_nuclei, indices_cyto_nuclei[cnt])
                    > self.find_overlap(
                        tmp_cyto, tmp_nuclei, indices_cyto_nuclei[cnt + 1]
                    )
                ) * 1
                del indices_cyto_nuclei[cnt + to_del]
                n = n - 1
            else:
                cnt = cnt + 1

        cyto_nuclei_ratio = [
            round(
                cyto_size[index_cyto_nuclei[0] - 1]
                / nuclei_size[index_cyto_nuclei[1] - 1],
                2,
            )
            for index_cyto_nuclei in indices_cyto_nuclei
        ]
        tmp_coords = [
            tuple(tmp_coords[index_cyto_nuclei[0] - 1])
            for index_cyto_nuclei in indices_cyto_nuclei
        ]
        return cyto_nuclei_ratio, tmp_coords, indices_cyto_nuclei

    def get_metrics(self, mask, custom_features, gui_self):
        """
        Calculate metrics for labeled regions in a binary mask.

            This method analyzes a binary image mask to extract features such as the
            size and roundness of labeled regions. It processes each region, filters
            out small regions, and computes the specified metrics based on custom
            features and GUI settings.

            Args:
                mask: A binary mask array representing the regions to analyze.
                custom_features: A set of features for measurement calculation.
                gui_self: An object containing GUI-related parameters, including flags
                           for calculating size and roundness, and a conversion factor
                           from pixels to millimeters.

            Returns:
                A tuple containing two lists and a list of center coordinates:
                    - A list with two elements: size_cells containing the sizes
                      of the detected regions, and round_cells containing their
                      roundness measures.
                    - A list of center coordinates for each detected region.
        """
        slices = ndimage.find_objects(mask.astype(int))
        center_coords = []
        size_cells = []
        round_cells = []

        for idx, si in enumerate(slices):
            mask_tmp = np.copy(mask).astype(np.uint8)
            mask_tmp[(idx + 1) != mask] = 0
            mask_tmp[(idx + 1) == mask] = 255

            padded_mask = np.pad(mask_tmp, 1, mode="constant")

            ####
            Zlabeled, Nlabels = ndimage.label(padded_mask)
            label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]

            # Remove the labels with size < 5
            for label, size in enumerate(label_size):
                if size < 5:
                    padded_mask[Zlabeled == label] = 0
            ####

            labels = dip.Label(padded_mask > 0)
            msr = dip.MeasurementTool.Measure(labels, features=custom_features)
            center_coords.append(
                [round(msr[1]["Center"][0], 2), round(msr[1]["Center"][1], 2)]
            )
            if gui_self.calcSize:
                size_cells.append(
                    round(msr[1]["SolidArea"][0] * pow(gui_self.px_to_mm, 2), 2)
                )
            if gui_self.calcRound:
                round_cells.append(round(msr[1]["Roundness"][0], 2))

        return [size_cells, round_cells], center_coords

    def out_concat(self, prev_out, curr_out):
        """
        Concatenates output values based on the type of previous output.

            This method checks the type of the `prev_out` parameter. If it is a float,
            it creates and returns a new list containing the previous output and the current output.
            If the `prev_out` is a list, it returns a new list that includes
            the elements of the previous list along with the current output.

            Args:
                prev_out: The previous output, which can either be a float or a list.
                curr_out: The current output, which is expected to be a value to be concatenated.

            Returns:
                A list containing the concatenated output values.
        """
        if isinstance(prev_out, float):
            return [prev_out, curr_out]
        else:  # isinstance(prev_out, list)
            return [prev_out[0], prev_out[1], curr_out]

    def get_voronoi_entropy(self, vor):
        """
        Calculate the entropy of the Voronoi diagram based on the classification of regions.

            This method computes the entropy of a Voronoi diagram by assessing the distribution of
            bounded regions in terms of their polygon class counts. It excludes unbounded regions
            from the calculations and uses the proportions of each polygon class to determine the
            overall entropy.

            Args:
                vor: An object representing the Voronoi diagram, which contains the
                     regions corresponding to each point and their vertices.

            Returns:
                A float representing the calculated Voronoi entropy, rounded to three decimal places.
        """
        polygon_class_counts = {}

        for region_index in vor.point_region:
            region_vertices = vor.regions[region_index]

            # Exclude unbounded regions
            if -1 not in region_vertices:
                polygon_class = len(region_vertices)

                if polygon_class in polygon_class_counts:
                    polygon_class_counts[polygon_class] += 1
                else:
                    polygon_class_counts[polygon_class] = 1

        # Total number of bounded regions and proportions
        total_bounded_regions = sum(polygon_class_counts.values())
        proportions = {
            polygon_class: count / total_bounded_regions
            for polygon_class, count in polygon_class_counts.items()
        }

        # Voronoi entropy
        voronoi_entropy = -sum(p * math.log(p) for p in proportions.values() if p > 0)
        return round(voronoi_entropy, 3)

    def save_metrics(
        self,
        masks_img,
        center_coords,
        metric_cells,
        metric_name,
        out_csv,
        out_name,
        out_dir,
        gui_self,
    ):
        """
        Saves various metrics and images derived from the input data to specified output paths.

            This method processes input mask images and metrics, generating colormap and labeled images,
            and outputs metrics to a CSV file. It combines existing metrics with new ones if provided.

            Args:
                masks_img: The image data containing mask information.
                center_coords: The coordinates of cell centers for labeling and metric calculation.
                metric_cells: The computed metric values for cells.
                metric_name: The name to assign to the saved metric image.
                out_csv: Existing CSV data to combine with new metrics.
                out_name: The name indicating the type of metrics being saved (e.g., 'size', 'roundness').
                out_dir: The directory where the output images and CSV files will be saved.
                gui_self: A reference to the GUI instance for file naming.

            Returns:
                A list containing the final metrics to be saved in the CSV.
        """
        layerz_cell = self.create_colormap_mask(masks_img)
        im_cell = Image.fromarray(layerz_cell)

        # Colormap img
        colormap_mask = Image.fromarray(self.create_colormap_mask(masks_img))
        im_masks = self.mask_indexing(colormap_mask, center_coords)
        im_masks.save(out_dir + "/" + "mask_colormap.png")

        # Metric img
        im_cell_size_labeled = self.image_labeling(
            im_mask=im_cell, im_labels=metric_cells, coords=center_coords
        )
        self.save_temp_output(
            image=im_cell_size_labeled, model_name=metric_name, gui_self=gui_self
        )
        im_cell_size_labeled.save(out_dir + "/" + metric_name + ".png")
        out_csv = (
            metric_cells
            if len(out_csv) == 0
            else [
                self.out_concat(out_csv[idx], single_metric)
                for idx, single_metric in enumerate(metric_cells)
            ]
        )

        # Metric csv
        header_title = []
        if out_name == "size":
            header_title = "area"
        elif out_name == "size_roundness":
            header_title = "area,roundness"
        elif out_name == "roundness":
            header_title = "roundness"
        elif out_name == "Center":
            header_title = "x,y"
        elif out_name == "ratio":
            header_title = "cell_id,nuclei_id,cell_nuclei_ratio"
        np.savetxt(
            out_dir
            + "/"
            + gui_self.filename.split("/")[-1].split(".")[0]
            + "_"
            + out_name
            + ".csv",
            out_csv,
            delimiter=", ",
            header=header_title,
            comments="",
            fmt="% s",
        )

        return out_csv

    def calculate_metrics(self, gui_self):
        """
        Calculate and save various metrics based on main and secondary mask images.

            This method computes metrics such as size, roundness, and ratio for main
            and secondary masks, and saves the results into corresponding directories.
            It can also generate Voronoi diagrams and calculate additional features
            such as convex hull area and Voronoi entropy based on user-defined options.

            Args:
                gui_self: An object containing the GUI information and options for
                          metric calculations, including options for size, roundness,
                          ratio calculations, and filename.

            Returns:
                A tuple containing:
                    - size_cells_main: The computed size metrics for the main mask.
                    - center_coords_main: The center coordinates of the main mask.
        """
        main_masks_img = self.temp_masks[self.indexCytoMask][
            1
        ]  # self.temp_masks[-1][1]
        secondary_masks_img = self.temp_masks[self.indexNucleusMask][1]
        comparison = main_masks_img == secondary_masks_img
        comparison = comparison.all()
        print("Are they equal?: ", comparison)

        # Create results dir
        results_dir = os.path.splitext(gui_self.filename)[0]
        primary_results_dir = results_dir + "/primary"
        secondary_results_dir = results_dir + "/secondary"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(primary_results_dir):
            os.makedirs(primary_results_dir)
        if not os.path.exists(secondary_results_dir):  # and not comparison:
            os.makedirs(secondary_results_dir)

        output_csv_primary = []
        output_csv_secondary = []
        output_csv_ratio = []
        output_csv_coords = []
        output_csv_voronoi = []

        output_name = ""

        # Set dip metrics
        custom_features = ["Center"]
        if gui_self.calcSize:
            custom_features.append("SolidArea")
            output_name += "size"
        if gui_self.calcRound:
            custom_features.append("Roundness")
            output_name += "_roundness"

        # Metrics for main mask
        main_metrics, center_coords_main = self.get_metrics(
            main_masks_img, custom_features, gui_self
        )
        size_cells_main, round_cells_main = main_metrics
        if gui_self.calcRatio:
            # Metrics for secondary mask (if exists)
            secondary_metrics, center_coords_secondary = self.get_metrics(
                secondary_masks_img, custom_features, gui_self
            )
            size_cells_secondary, round_cells_secondary = secondary_metrics
            ratio_cells, center_coords_ratio, indices_cyto_nuclei = (
                self.matched_indices(
                    main_masks_img,
                    secondary_masks_img,
                    size_cells_main,
                    size_cells_secondary,
                    center_coords_main,
                )
            )
            output_csv_ratio = [
                [index_cyto_nuclei[0], index_cyto_nuclei[1]]
                for index_cyto_nuclei in indices_cyto_nuclei
            ]

        for idx_feature, feature in enumerate(custom_features):
            if idx_feature > 0:
                output_csv_primary = self.save_metrics(
                    main_masks_img,
                    center_coords_main,
                    main_metrics[idx_feature - 1],
                    custom_features[idx_feature],
                    output_csv_primary,
                    output_name,
                    primary_results_dir,
                    gui_self,
                )

                if gui_self.calcRatio:
                    output_csv_secondary = self.save_metrics(
                        secondary_masks_img,
                        center_coords_secondary,
                        secondary_metrics[idx_feature - 1],
                        custom_features[idx_feature],
                        output_csv_secondary,
                        output_name,
                        secondary_results_dir,
                        gui_self,
                    )

        if gui_self.calcRatio:
            output_csv_ratio = self.save_metrics(
                main_masks_img,
                center_coords_ratio,
                ratio_cells,
                "ratio",
                output_csv_ratio,
                "ratio",
                results_dir,
                gui_self,
            )

        output_csv_coords = self.save_metrics(
            main_masks_img,
            center_coords_main,
            center_coords_main,
            "Center",
            output_csv_coords,
            "Center",
            primary_results_dir,
            gui_self,
        )

        if gui_self.calcVoronoi:
            img = plt.imread(gui_self.filename)

            output_csv_coords = pd.DataFrame(output_csv_coords)
            output_csv_coords[1] = img.shape[0] - output_csv_coords[1]

            vor = Voronoi(output_csv_coords)
            fig = voronoi_plot_2d(vor)

            fig, ax = plt.subplots()
            ax.imshow(ndimage.rotate(np.fliplr(img), 180))
            fig = voronoi_plot_2d(vor, point_size=10, ax=ax, line_colors="red")
            plt.savefig(results_dir + "/" + "voronoi.png")

            ### Convex Hull

            conhull = ConvexHull(output_csv_coords)
            fig = convex_hull_plot_2d(conhull)

            fig, ax = plt.subplots()
            ax.imshow(ndimage.rotate(np.fliplr(img), 180))
            fig = convex_hull_plot_2d(conhull, ax=ax)
            plt.savefig(results_dir + "/" + "hull.png")

            conhull_area = conhull.area
            np.savetxt(
                results_dir
                + "/"
                + gui_self.filename.split("/")[-1].split(".")[0]
                + "_convex_hull_area.csv",
                [round(conhull_area, 3)],
                delimiter=", ",
                header="convex_hull_area",
                comments="",
                fmt="% s",
            )
            ###

            voronoi_entropy = self.get_voronoi_entropy(vor)
            np.savetxt(
                results_dir
                + "/"
                + gui_self.filename.split("/")[-1].split(".")[0]
                + "_vornoi_entropy.csv",
                [voronoi_entropy],
                delimiter=", ",
                header="voronoi_entropy",
                comments="",
                fmt="% s",
            )

            CSM_array = symmetry.CSM_for_graph(vor)
            np.savetxt(
                results_dir
                + "/"
                + gui_self.filename.split("/")[-1].split(".")[0]
                + "_CSM_values.csv",
                [round(np.asarray(CSM_array).mean(), 3)],
                delimiter=", ",
                header="CSM_array",
                comments="",
                fmt="% s",
            )

        return size_cells_main, center_coords_main

    def select_mask(self, menu_output, cell_type, curr_index, gui_self):
        """
        Selects a mask in the given menu based on the cell type and index.

            This method updates the mask selection for either primary or secondary cell types.
            It adjusts the icons of previously selected masks, sets the current mask based on
            the provided index, and enables or disables related GUI checkboxes and buttons
            based on the current selection.

            Args:
                menu_output: The output menu where the mask is selected from.
                cell_type: The type of cell, which can be "primary" or "secondary".
                curr_index: The index of the current selection in the menu.
                gui_self: The GUI instance that holds references to various UI elements
                          such as checkboxes and buttons.

            Returns:
                None: This method does not return a value, but modifies the GUI state
                based on the current selections.
        """
        if cell_type == "primary":
            if self.indexCytoMask != -1:
                prev_selected_mask = menu_output.parentWidget().findChildren(QMenu)[
                    self.indexCytoMask
                ]
                prev_selected_mask.setIcon(QtGui.QIcon())
            self.indexCytoMask = curr_index
            self.indexNucleusMask = (
                -1 if self.indexNucleusMask == curr_index else self.indexNucleusMask
            )
        elif cell_type == "secondary":
            if self.indexNucleusMask != -1:
                prev_selected_mask = menu_output.parentWidget().findChildren(QMenu)[
                    self.indexNucleusMask
                ]
                prev_selected_mask.setIcon(QtGui.QIcon())
            self.indexNucleusMask = curr_index
            self.indexCytoMask = (
                -1 if self.indexCytoMask == curr_index else self.indexCytoMask
            )
        icon_path = pathlib.Path.home().joinpath(".cellpose", str(cell_type) + ".png")
        menu_output.setIcon(QtGui.QIcon(str(icon_path.resolve())))

        gui_self.RTCheckBox.setEnabled(
            self.indexCytoMask > -1 and self.indexNucleusMask > -1
        )
        gui_self.VDCheckBox.setEnabled(
            self.indexCytoMask > -1 and self.indexNucleusMask > -1
        )
        gui_self.SMCheckBox.setEnabled(self.indexCytoMask > -1)
        gui_self.RMCheckBox.setEnabled(self.indexCytoMask > -1)
        # self.CalculateButton.setStyleSheet(self.styleUnpressed if self.indexCytoMask > -1 else self.styleInactive)
        gui_self.CalculateButton.setEnabled(self.indexCytoMask > -1)

    def select_image(self, gui_self, img_layer, name):
        """
        Selects and displays an image layer in the GUI.

            This method checks if the currently selected image mask is different from the provided name.
            If they are different, it sets the specified image layer in the GUI. If they are the same,
            it updates the layer and clears the current image mask.

            Args:
                gui_self: The GUI instance that manages the image display.
                img_layer: The image layer to be displayed.
                name: The name of the image mask to check against the currently selected image.

            Returns:
                None
        """
        if self.currentImageMask != name:
            gui_self.layer.setImage(np.asarray(img_layer), autoLevels=False)
            self.currentImageMask = name
        else:
            self.update_layer()
            self.currentImageMask = ""
        print("WEEEEEE 3")
