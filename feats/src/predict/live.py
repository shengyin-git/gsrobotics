#!/usr/bin/env python3

import sys
from pathlib import Path
import os
import argparse
import yaml

# Ensure imports resolve regardless of current working directory.
# Make the repository's `src` directory available on sys.path.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from data.dataloader import normalize, unnormalize
from models.unet import UNet

# Import Reconstruction3D for higher-fidelity depth and contact maps
# Ensure repo root is on sys.path for utilities
repo_root = str(SCRIPT_DIR.parent.parent.parent)  # feats/src/predict -> feats/src -> feats -> repo
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utilities.reconstruction import Reconstruction3D
from utilities.image_processing import trim_outliers, normalize_array, apply_cmap, color_map_from_txt

## to do list:
## 1. implement force zeroing by considering distributions instead of simple averaging
## 2. implement force center and magnitude visualization on the plots for maximum forces
## 3. implement better visualization for depth and contact mask using reconstruction3D class
## 4. implement calibration option for reconstruction3D class
## 5. clean up the code

def capture_image(cam, imgw=320, imgh=240):
    """
    Capture image from GelSight Mini and process it.

    :param cam: video capture object
    :param imgw: width of the image
    :param imgh: height of the image
    :return: processed image
    """

    # capture image
    ret, f0 = cam.read()

    if ret:
        # resize, crop and resize back
        img = cv2.resize(f0, (895, 672))  # size suggested by janos to maintain aspect ratio
        border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
        img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
        img = img[:, :-1]  # remove last column to get a popular image resolution
        img = cv2.resize(img, (imgw, imgh))  # final resize for 3d
    else:
        print("ERROR! reading image from camera")

    # convert bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# def reform_img_feats(f0, imgw=320, imgh=240):
#     """
#     Capture image from GelSight Mini and process it.

#     :param f0
#     :param imgw: width of the image
#     :param imgh: height of the image
#     :return: processed image
#     """

#     # resize, crop and resize back
#     img = cv2.resize(f0, (895, 672))  # size suggested by janos to maintain aspect ratio
#     border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
#     img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
#     img = img[:, :-1]  # remove last column to get a popular image resolution
#     img = cv2.resize(img, (imgw, imgh))  # final resize for 3d

#     # convert bgr to rgb
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     return img


# def reform_img_depth(f0, imgw=320, imgh=240):
#     """
#     Capture image from GelSight Mini and process it.

#     :param f0
#     :param imgw: width of the image
#     :param imgh: height of the image
#     :return: processed image
#     """

#     # resize, crop and resize back
#     img = cv2.resize(f0, (895, 672))  # size suggested by janos to maintain aspect ratio
#     border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
#     img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
#     img = img[:, :-1]  # remove last column to get a popular image resolution
#     img = cv2.resize(img, (imgw, imgh))  # final resize for 3d

#     # convert bgr to rgb
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     return img

class force_prediction:
    def __init__(self):
        self.force_zero_counter = 0
        self.pred_grid_x_zero, self.pred_grid_y_zero, self.pred_grid_z_zero = None, None, None
    
    def make_prediction(self, img, model, device, config):
        """
        Make prediction using the unet model.

        :param img: input image
        :param model: unet model
        :param device: device to run the model
        :param config: configuration file
        :return: predicted grid x, y, z
        """

        # store data in dictionary
        data = {}
        data["gs_img"] = img

        # normalize data
        data = normalize(data, config["norm_file"])

        # convert to torch tensor
        gs_img = torch.from_numpy(data["gs_img"]).float()
        gs_img = gs_img.unsqueeze(0).permute(0, 3, 1, 2).to(device)

        # load calibration file
        if config["calibration_file"] is not None:
            calibration = np.load(config["calibration_file"])
            rows, cols = 240, 320
            M = np.float32([[1, 0, calibration[0]], [0, 1, calibration[1]]])

        # prepare input data
        if config["calibration_file"] is not None:
            inputs_prewarp = data["gs_img"]
            inputs_warp = cv2.warpAffine(inputs_prewarp,  M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            inputs = torch.from_numpy(inputs_warp).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            inputs = gs_img

        # get model prediction
        outputs = model(inputs)

        # unnormalize the outputs
        outputs_transf = outputs.squeeze(0).permute(1, 2, 0)
        pred_grid_x = unnormalize(outputs_transf[:, :, 0], "grid_x", config["norm_file"])
        pred_grid_y = unnormalize(outputs_transf[:, :, 1], "grid_y", config["norm_file"])
        pred_grid_z = unnormalize(outputs_transf[:, :, 2], "grid_z", config["norm_file"])

        # convert to numpy
        pred_grid_x = pred_grid_x.cpu().detach().numpy()
        pred_grid_y = pred_grid_y.cpu().detach().numpy()
        pred_grid_z = pred_grid_z.cpu().detach().numpy()

        if self.force_zero_counter < 50:

            # Update zero force for the first 50 frames.
            if self.force_zero_counter == 0:
                self.pred_grid_x_zero = pred_grid_x
                self.pred_grid_y_zero = pred_grid_y
                self.pred_grid_z_zero = pred_grid_z
                print("Zeroing force. Please do not touch the sensor...")
            else:
                self.pred_grid_x_zero += pred_grid_x
                self.pred_grid_y_zero += pred_grid_y
                self.pred_grid_z_zero += pred_grid_z

            if self.force_zero_counter == 49:
                self.pred_grid_x_zero /= (
                    self.force_zero_counter + 1
                )  # +1 to include current frame
                self.pred_grid_y_zero /= (
                    self.force_zero_counter + 1
                )
                self.pred_grid_z_zero /= (
                    self.force_zero_counter + 1
                )

        if self.force_zero_counter == 50:
            print("Force prediction is ready to use.")

        self.force_zero_counter += 1

        # Subtract the accumulated zero depth for normalization.
        pred_grid_x -= self.pred_grid_x_zero
        pred_grid_y -= self.pred_grid_y_zero
        pred_grid_z -= self.pred_grid_z_zero

        return pred_grid_x, pred_grid_y, pred_grid_z
    

    def make_prediction_ori(self, img, model, device, config):
        """
        Make prediction using the unet model.

        :param img: input image
        :param model: unet model
        :param device: device to run the model
        :param config: configuration file
        :return: predicted grid x, y, z
        """

        # store data in dictionary
        data = {}
        data["gs_img"] = img

        # normalize data
        data = normalize(data, config["norm_file"])

        # convert to torch tensor
        gs_img = torch.from_numpy(data["gs_img"]).float()
        gs_img = gs_img.unsqueeze(0).permute(0, 3, 1, 2).to(device)

        # load calibration file
        if config["calibration_file"] is not None:
            calibration = np.load(config["calibration_file"])
            rows, cols = 240, 320
            M = np.float32([[1, 0, calibration[0]], [0, 1, calibration[1]]])

        # prepare input data
        if config["calibration_file"] is not None:
            inputs_prewarp = data["gs_img"]
            inputs_warp = cv2.warpAffine(inputs_prewarp,  M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            inputs = torch.from_numpy(inputs_warp).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            inputs = gs_img

        # get model prediction
        outputs = model(inputs)

        # unnormalize the outputs
        outputs_transf = outputs.squeeze(0).permute(1, 2, 0)
        pred_grid_x = unnormalize(outputs_transf[:, :, 0], "grid_x", config["norm_file"])
        pred_grid_y = unnormalize(outputs_transf[:, :, 1], "grid_y", config["norm_file"])
        pred_grid_z = unnormalize(outputs_transf[:, :, 2], "grid_z", config["norm_file"])

        # convert to numpy
        pred_grid_x = pred_grid_x.cpu().detach().numpy()
        pred_grid_y = pred_grid_y.cpu().detach().numpy()
        pred_grid_z = pred_grid_z.cpu().detach().numpy()

        return pred_grid_x, pred_grid_y, pred_grid_z


def compute_force_center_and_magnitude(force_grid):
    """
    Compute the center of force and total magnitude for a 2D force grid.

    :param force_grid: 2D array of force values
    :return: (center_y, center_x, magnitude)
    """
    # Normalize force grid to weights (absolute values)
    weights = np.abs(force_grid)
    total_weight = np.sum(weights)
    
    if total_weight < 1e-6:
        # No force; return center of grid
        center_y = force_grid.shape[0] / 2.0
        center_x = force_grid.shape[1] / 2.0
        magnitude = 0.0
    else:
        # Compute weighted center
        y_indices, x_indices = np.meshgrid(np.arange(force_grid.shape[0]), 
                                            np.arange(force_grid.shape[1]), 
                                            indexing='ij')
        center_y = np.sum(y_indices * weights) / total_weight
        center_x = np.sum(x_indices * weights) / total_weight
        magnitude = np.sum(force_grid)
    
    return center_y, center_x, magnitude


def animate(frame, cam, model, device, config, ims, axs, overlay_artists, persistent):
    """
    Animate the prediction with matplotlib.

    :param frame: frame number
    :param cam: camera object
    :param mdoel: unet model
    :param device: device to run the model
    :param config: configuration dictionary
    :param ims: image objects
    :param axs: axis objects
    :param overlay_artists: dict to store overlay artists for removal
    :return: None
    """

    # capture image and make prediction

    gs_img = capture_image(cam)
    pred_grid_x, pred_grid_y, pred_grid_z = persistent['force_pred'].make_prediction(gs_img, model, device, config)

    # update the image data
    ims[0].set_data(pred_grid_x)
    ims[1].set_data(pred_grid_y)
    ims[2].set_data(pred_grid_z)
    ims[3].set_data(gs_img.astype(np.uint8))

    # Compute force centers and magnitudes
    center_y_x, center_x_x, mag_x = compute_force_center_and_magnitude(pred_grid_x)
    center_y_y, center_x_y, mag_y = compute_force_center_and_magnitude(pred_grid_y)
    center_y_z, center_x_z, mag_z = compute_force_center_and_magnitude(pred_grid_z)

    # Clear previous overlays by removing stored artists
    for ax_idx in range(3):
        if ax_idx in overlay_artists:
            for artist in overlay_artists[ax_idx]:
                artist.remove()
        overlay_artists[ax_idx] = []

    # Overlay X-force: center point + arrow representing magnitude
    arrow_scale_x = 2.0  # Scale arrow length by this factor
    pt_x = axs[0].plot(center_x_x, center_y_x, 'r+', markersize=15, markeredgewidth=2, clip_on=True)
    overlay_artists[0].extend(pt_x)
    if abs(mag_x) > 1e-6:
        # Arrow pointing in direction of force (left/right based on sign)
        dx = np.sign(mag_x) * arrow_scale_x
        arrow_x = axs[0].arrow(center_x_x, center_y_x, dx, 0, head_width=1.5, head_length=1.0, 
                     fc='red', ec='red', alpha=0.7, clip_on=True)
        overlay_artists[0].append(arrow_x)

    # Overlay Y-force: center point + arrow representing magnitude
    arrow_scale_y = 2.0
    pt_y = axs[1].plot(center_x_y, center_y_y, 'g+', markersize=15, markeredgewidth=2, clip_on=True)
    overlay_artists[1].extend(pt_y)
    if abs(mag_y) > 1e-6:
        # Arrow pointing in direction of force (up/down based on sign)
        dy = np.sign(mag_y) * arrow_scale_y
        arrow_y = axs[1].arrow(center_x_y, center_y_y, 0, dy, head_width=1.5, head_length=1.0, 
                     fc='green', ec='green', alpha=0.7, clip_on=True)
        overlay_artists[1].append(arrow_y)

    # (X/Y row-mean curves removed — only Z subplot shows row-mean curve)

    # Overlay Z-force: circle at center, radius representing magnitude
    circle_scale_z = 0.5  # Scale circle radius by this factor (smaller to keep image size fixed)
    radius_z = abs(mag_z) * circle_scale_z
    circle = plt.Circle((center_x_z, center_y_z), radius_z, fill=False, 
                         edgecolor='blue', linewidth=2, alpha=0.7)
    circle.set_clip_on(True)
    axs[2].add_patch(circle)
    overlay_artists[2].append(circle)
    
    # Add fixed-size cross at force center (match X/Y markers)
    pt_z = axs[2].plot(center_x_z, center_y_z, 'b+', markersize=15, markeredgewidth=2, clip_on=True)
    overlay_artists[2].extend(pt_z)

    # Update contact mask and depth panels using Reconstruction3D
    reconstruction = persistent['reconstruction']
    gs_img_depth = cv2.cvtColor(gs_img, cv2.COLOR_BGR2RGB)
    # Get high-quality depth map and contact mask from Reconstruction3D
    depth_map, contact_mask, grad_x, grad_y = reconstruction.get_depthmap(
        image=gs_img_depth,
        markers_threshold=(config.get('marker_mask_min', 0), 
                            config.get('marker_mask_max', 70))
    )
    
    # Check for NaN values in depth map
    if np.isnan(depth_map).any():
        raise ValueError("Depth map contains NaN values")
    
    # Process depth map: trim outliers, normalize, and apply colormap
    depth_map_trimmed = trim_outliers(depth_map, 1, 99)
    depth_map_normalized = normalize_array(array=depth_map_trimmed, min_divider=10)
    
    depth_rgb = apply_cmap(data=depth_map_normalized, cmap=persistent['cmap'])
    # apply_cmap returns RGB, convert to 0-1 range for imshow
    ims[5].set_data(depth_rgb.astype(np.uint8))
    
    # Process contact mask: convert to 8-bit grayscale (0-255), then to RGB for display
    contact_display = (contact_mask * 255).astype(np.uint8)
    # Convert grayscale to RGB so it displays properly with the existing colormap
    contact_rgb = cv2.cvtColor(contact_display, cv2.COLOR_GRAY2RGB)
    # Normalize to 0-1 for imshow
    ims[4].set_data(contact_rgb.astype(np.uint8))

    # Compute per-row force center (x center for each row) and update persistent line
    n_rows, n_cols = pred_grid_z.shape
    row_centers = np.zeros(n_rows)
    cols_idx = np.arange(n_cols)
    for i in range(n_rows):
        row = pred_grid_z[i, :]
        w = np.abs(row)
        s = np.sum(w)
        if s < 1e-6:
            row_centers[i] = (n_cols - 1) / 2.0
        else:
            row_centers[i] = np.sum(cols_idx * w) / s

    x_positions = row_centers
    y_positions = np.arange(n_rows)

    if 'z_row_line' in persistent and persistent['z_row_line'] is not None:
        persistent['z_row_line'].set_data(x_positions, y_positions)
    else:
        line, = axs[2].plot(x_positions, y_positions, color='magenta', linewidth=2, alpha=0.9)
        persistent['z_row_line'] = line

    # Compute per-column force center (y center for each column) and update persistent line
    col_centers = np.zeros(n_cols)
    rows_idx = np.arange(n_rows)
    for j in range(n_cols):
        col = pred_grid_z[:, j]
        w = np.abs(col)
        s = np.sum(w)
        if s < 1e-6:
            col_centers[j] = (n_rows - 1) / 2.0
        else:
            col_centers[j] = np.sum(rows_idx * w) / s

    x_positions_col = np.arange(n_cols)
    y_positions_col = col_centers

    if 'z_col_line' in persistent and persistent['z_col_line'] is not None:
        persistent['z_col_line'].set_data(x_positions_col, y_positions_col)
    else:
        col_line, = axs[2].plot(x_positions_col, y_positions_col, color='cyan', linewidth=2, alpha=0.9)
        persistent['z_col_line'] = col_line

    # update titles with the new force values
    axs[0].set_title("x-force: {:.2f}N".format(np.sum(pred_grid_x)), fontsize=14)
    axs[1].set_title("y-force: {:.2f}N".format(np.sum(pred_grid_y)), fontsize=14)
    axs[2].set_title("z-force: {:.2f}N".format(np.sum(pred_grid_z)), fontsize=14)

    return None


def main(config):
    """
    Main function to run the live prediction.

    :param config: configuration dictionary containing model parameters and paths
    :return: None
    """

    # initialize camera
    cam = cv2.VideoCapture(4) # 0 4
    img_height, img_width = 240, 320

    # specify device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = UNet(enc_chs=config["enc_chs"], dec_chs=config["dec_chs"], out_sz=config["output_size"])
    model.load_state_dict(torch.load(config["model"], map_location=torch.device("cpu"), weights_only=True))
    model.eval().to(device)

    # Set up the figure and initial images (2 rows x 3 columns)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    # flatten axes array so existing code can index axs[0], axs[1], ...
    axs = axs.flatten()
    fig.canvas.manager.set_window_title("FEATS")
    fig.suptitle("\nFEATS LIVE DEMO", fontsize=16)

    # reduce whitespace of plot
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)

    # capture first image of the camera
    gs_img = capture_image(cam, imgw=img_width, imgh=img_height)

    # make initial prediction
    force_pred = force_prediction()
    pred_grid_x, pred_grid_y, pred_grid_z = force_pred.make_prediction(gs_img, model, device, config)

    # hardcode the color limits
    clim_x = (-0.029, 0.029)
    clim_y = (-0.029, 0.029)
    clim_z = (-0.17, 0.0)

    # initialize the plots
    im_x = axs[0].imshow(pred_grid_x, origin="upper", vmin=clim_x[0], vmax=clim_x[1], animated=True)
    im_y = axs[1].imshow(pred_grid_y, origin="upper", vmin=clim_y[0], vmax=clim_y[1], animated=True)
    im_z = axs[2].imshow(pred_grid_z, origin="upper", vmin=clim_z[0], vmax=clim_z[1], animated=True)
    im_gs = axs[3].imshow(gs_img.astype(np.uint8), animated=True)

    # Pre-create persistent artists (Z row- and column-center lines) to update each frame
    persistent = { 'z_row_line': None, 'z_col_line': None }
    persistent['force_pred'] = force_pred
    # create empty Line2D objects on axs[2] and store them
    persistent['z_row_line'], = axs[2].plot([], [], color='magenta', linewidth=2, alpha=0.9)
    persistent['z_col_line'], = axs[2].plot([], [], color='cyan', linewidth=2, alpha=0.9)
    # ensure persistent lines are clipped to the axes
    persistent['z_row_line'].set_clip_on(True)
    persistent['z_col_line'].set_clip_on(True)

    # load depth prediction model
    # initialize Reconstruction3D
    reconstruction = Reconstruction3D(
        image_width=img_width,
        image_height=img_height,
        use_gpu=torch.cuda.is_available()
    )
    
    # Resolve nn_model_path relative to the actual repo root (not feats copy)
    # Since feats is a copy, we need to look in the original repo root
    nn_model_path = config.get("nn_model_path", "./models/nnmini.pt")
    
    # Try multiple possible locations
    possible_roots = [
        Path.cwd(),  # Current working directory (should be repo root when run from there)
        Path(__file__).resolve().parent.parent.parent,  # feats/src/predict -> feats
        Path(__file__).resolve().parent.parent.parent.parent,  # feats/src/predict -> repo root
    ]
    
    resolved_path = None
    for root in possible_roots:
        candidate = (root / nn_model_path).resolve()
        if candidate.exists():
            resolved_path = str(candidate)
            break
    
    if resolved_path is None:
        raise FileNotFoundError(f"NN model not found. Tried: {[str((r / nn_model_path).resolve()) for r in possible_roots]}")
    
    print(f"Loading NN model from: {resolved_path}")
    reconstruction.load_nn(resolved_path)
    persistent['reconstruction'] = reconstruction
    print("NN model loaded successfully.")

    # Load colormap for depth visualization
    cmap_path = config.get("cmap_txt_path", "./cmap.txt")
    
    # Try multiple possible locations
    possible_roots = [
        Path.cwd(),  # Current working directory
        Path(__file__).resolve().parent.parent.parent,  # feats/src/predict -> feats
        Path(__file__).resolve().parent.parent.parent.parent,  # feats/src/predict -> repo root
    ]
    
    resolved_path = None
    for root in possible_roots:
        candidate = (root / cmap_path).resolve()
        if candidate.exists():
            resolved_path = str(candidate)
            break
    
    if resolved_path is None:
        raise FileNotFoundError(f"Colormap not found. Tried: {[str((r / cmap_path).resolve()) for r in possible_roots]}")
    
    print(f"Loading colormap from: {resolved_path}")
    persistent['cmap'] = color_map_from_txt(path=resolved_path, is_bgr=config.get("cmap_in_BGR_format", True))
    print("Colormap loaded successfully.")

    # Add marker threshold settings to persistent dict
    persistent['marker_mask_min'] = config.get("marker_mask_min", 0)
    persistent['marker_mask_max'] = config.get("marker_mask_max", 70)

    gs_img_depth = cv2.cvtColor(gs_img, cv2.COLOR_BGR2RGB)
    depth_map, contact_mask, grad_x, grad_y = reconstruction.get_depthmap(
                image=gs_img_depth,
                markers_threshold=(config.get('marker_mask_min', 0), 
                                  config.get('marker_mask_max', 70))
            )
    
    # Check for NaN values in depth map
    if np.isnan(depth_map).any():
        raise ValueError("Depth map contains NaN values")
    
    # Process depth map: trim outliers, normalize, and apply colormap
    depth_map_trimmed = trim_outliers(depth_map, 1, 99)
    depth_map_normalized = normalize_array(array=depth_map_trimmed, min_divider=10)
    
    depth_rgb = apply_cmap(data=depth_map_normalized, cmap=persistent['cmap'])
    # apply_cmap returns RGB, convert to 0-1 range for imshow
    im_depth = axs[5].imshow(depth_rgb.astype(np.uint8), animated=True)

    # Process contact mask: convert to 8-bit grayscale (0-255), then to RGB for display
    contact_display = (contact_mask * 255).astype(np.uint8)
    # Convert grayscale to RGB so it displays properly with the existing colormap
    contact_rgb = cv2.cvtColor(contact_display, cv2.COLOR_GRAY2RGB)
    # Normalize to 0-1 for imshow
    im_contact = axs[4].imshow(contact_rgb.astype(np.uint8), animated=True)

    # store the images in a list (order: x,y,z,original,contact,depth)
    ims = [im_x, im_y, im_z, im_gs, im_contact, im_depth]

    # Fix axis limits for the first and second subplots so their size and
    # position remain constant even if overlay artists extend outside.
    try:
        n_rows, n_cols = pred_grid_x.shape
    except Exception:
        # fallback to a default grid size
        n_rows, n_cols = 24, 32
    # Set limits to pixel coordinates (imshow uses center at 0..n-1)
    for ax in (axs[0], axs[1]):
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_autoscale_on(False)
    # Also fix axis limits for the Z subplot so it remains the same size/position
    try:
        n_rows_z, n_cols_z = pred_grid_z.shape
    except Exception:
        n_rows_z, n_cols_z = n_rows, n_cols
    axs[2].set_xlim(-0.5, n_cols_z - 0.5)
    axs[2].set_ylim(n_rows_z - 0.5, -0.5)
    axs[2].set_autoscale_on(False)

    # set titles
    axs[0].set_title("x-force: {:.2f}N".format(np.sum(pred_grid_x)), fontsize=14)
    axs[1].set_title("y-force: {:.2f}N".format(np.sum(pred_grid_y)), fontsize=14)
    axs[2].set_title("z-force: {:.2f}N".format(np.sum(pred_grid_z)), fontsize=14)
    axs[3].set_title("GelSight Mini Image", fontsize=14)
    axs[4].set_title("Contact Mask", fontsize=14)
    axs[5].set_title("Depth (pred)", fontsize=14)

    # turn off axis for all six subplots
    for ax in axs:
        ax.axis("off")

    # show colorbars
    fig.colorbar(im_x, ax=axs[0], orientation="horizontal", fraction=0.046, pad=0.01, ticks=[-0.02, -0.01, 0.0, 0.01, 0.02])
    fig.colorbar(im_y, ax=axs[1], orientation="horizontal", fraction=0.046, pad=0.01, ticks=[-0.02, -0.01, 0.0, 0.01, 0.02])
    fig.colorbar(im_z, ax=axs[2], orientation="horizontal", fraction=0.046, pad=0.01, ticks=[-0.15, -0.1, -0.05, 0.0])

    # add fake colorbar for axs[3] so that gs_img is on the same height
    fake_cbar = fig.colorbar(im_gs, ax=axs[3], orientation="horizontal", fraction=0.046, pad=0.01)
    fake_cbar.ax.set_visible(False)

    # Initialize overlay artists dictionary for frame updates
    overlay_artists = {0: [], 1: [], 2: []}

    # animate the predictions
    ani = FuncAnimation(fig, animate, frames=None, interval=0, blit=False, fargs=(cam, model, device, config, ims, axs, overlay_artists, persistent), save_count=100)

    plt.show()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Live prediction of UNet")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file for live prediction.")
    args = parser.parse_args()

    # Determine config path: default to the predict directory's config when not provided
    if args.config is None:
        config_path = SCRIPT_DIR / "predict_config.yaml"
    else:
        config_path = Path(args.config)

    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    # load config file
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # Resolve relative config paths (model, norm_file, calibration_file) relative to config file
    cfg_dir = config_path.parent
    for key in ("model", "norm_file", "calibration_file"):
        if key in config and config[key] is not None:
            val = config[key]
            if not os.path.isabs(val):
                config[key] = str((cfg_dir / val).resolve())

    # make live prediction
    main(config)
