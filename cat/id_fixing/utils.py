import numpy as np
import os
import subprocess
import glob
import cv2
import shutil  # <-- needed for copying
import yaml
import parmap
from datetime import datetime, timezone, timedelta
import pandas as pd
import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import cv2




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
yolo_utils.py
-------------
Utility functions for loading YOLO tracking CSVs, building 4D track arrays,
matching tracks across frames, and plotting trajectories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣  Data loading and array construction
# ------------------------------
def load_yolo_csv(df, n_tracks=10, n_frames=1500):
    """
    Convert a YOLO tracking DataFrame into a 4D array [track, frame, feature, (x, y, p)].

    Parameters
    ----------
    df : pd.DataFrame
        YOLO CSV containing FRAME, TRACK, and feature (X,Y,P) columns.
    n_tracks : int, optional
        Maximum number of tracks to allocate (default 10).
    n_frames : int, optional
        Fixed number of frames per video (default 1500).

    Returns
    -------
    np.ndarray
        4D array of shape (n_tracks, n_frames, n_features, 3).
    """
    features = ["NOSE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SIDE",
                "CENTER", "RIGHT_SIDE", "TAIL_BASE"]
    n_features = len(features)
    tracks = np.full((n_tracks, n_frames, n_features, 3), np.nan)

    for _, row in df.iterrows():
        frame = int(row["FRAME"])
        track = int(row["TRACK"])
        if track >= n_tracks or frame >= n_frames:
            continue

        for f_idx, f_name in enumerate(features):
            x = row[f"{f_name}_X"]
            y = row[f"{f_name}_Y"]
            p = row[f"{f_name}_P"]
            tracks[track - 1, frame, f_idx, :] = [x, y, p]

    return tracks


# ------------------------------
# 2️⃣  Matching across frames
# ------------------------------
def match_tracks_greedy(tracks, n_animals=3, max_dist=100):
    """
    Greedy nearest-neighbor track matching across frames.
    Maintains n_animals continuous tracks, filling missing data with previous positions.

    Parameters
    ----------
    tracks : np.ndarray
        (n_tracks_detected, n_frames, n_features, 3)
    n_animals : int
        True number of tracked animals (default 3)
    max_dist : float
        Maximum distance (in px) to consider a valid match.

    Returns
    -------
    np.ndarray
        Matched track array of shape (n_animals, n_frames, n_features, 3)
    """
    n_tracks_detected, n_frames, n_features, _ = tracks.shape
    matched_tracks = np.full((n_animals, n_frames, n_features, 3), np.nan)

    def compute_centroid(frame_tracks):
        return np.nanmedian(frame_tracks[:, :, :2], axis=1)  # (n_tracks, 2)

    # --- initialize from first frame ---
    centroids_0 = compute_centroid(tracks[:, 0])
    valid_0 = ~np.isnan(centroids_0[:, 0])
    valid_indices = np.where(valid_0)[0][:n_animals]
    for i, idx in enumerate(valid_indices):
        matched_tracks[i, 0, :, :] = tracks[idx, 0, :, :]

    # --- main greedy matching loop ---
    for t in range(1, n_frames):
        prev_centroids = compute_centroid(matched_tracks[:, t - 1])
        curr_centroids = compute_centroid(tracks[:, t])
        curr_valid = ~np.isnan(curr_centroids[:, 0])
        curr_pts = curr_centroids[curr_valid]
        curr_ids = np.where(curr_valid)[0]

        if len(curr_pts) == 0:
            matched_tracks[:, t, :, :] = matched_tracks[:, t - 1, :, :]
            continue

        dist = np.linalg.norm(prev_centroids[:, None, :] - curr_pts[None, :, :], axis=2)
        used_curr = set()

        for i in range(n_animals):
            nearest_idx = np.argmin(dist[i])
            c_idx = curr_ids[nearest_idx]
            d = dist[i, nearest_idx]

            if d > max_dist or nearest_idx in used_curr:
                matched_tracks[i, t, :, :] = matched_tracks[i, t - 1, :, :]
            else:
                matched_tracks[i, t, :, :] = tracks[c_idx, t, :, :]
                used_curr.add(nearest_idx)

        # fill missing
        for i in range(n_animals):
            if np.isnan(matched_tracks[i, t, 0, 0]):
                matched_tracks[i, t, :, :] = matched_tracks[i, t - 1, :, :]

    return matched_tracks


def compute_centroid_over_time2(all_tracks: np.ndarray) -> np.ndarray:
    """
    Compute the centroid (median x,y) over time for all animals in one array,
    with forward fill for missing frames per animal.

    Parameters
    ----------
    all_tracks : np.ndarray
        Array of shape (n_animals, n_frames, n_features, 3),
        where the last dimension is (x, y, p).

    Returns
    -------
    np.ndarray
        Centroid positions of shape (n_animals, n_frames, 2)
    """
    n_animals, n_frames, _, _ = all_tracks.shape
    centroids = np.nanmedian(all_tracks[:, :, :, :2], axis=2)  # (n_animals, n_frames, 2)

    # forward fill NaNs per animal
    for i in range(n_animals):
        for t in range(1, n_frames):
            if np.isnan(centroids[i, t, 0]) or np.isnan(centroids[i, t, 1]):
                centroids[i, t] = centroids[i, t - 1]

    return centroids
# ------------------------------
# 3️⃣  Centroid computation
# ------------------------------
def compute_centroid_over_time(frame_tracks):
    """
    Compute centroid (x,y) over time for one track, with forward fill for missing frames.
    """
    centroid = np.nanmedian(frame_tracks[:, :, :2], axis=1)
    for i in range(1, centroid.shape[0]):
        if np.isnan(centroid[i, 0]) or np.isnan(centroid[i, 1]):
            centroid[i] = centroid[i - 1]
    return centroid
from tqdm import tqdm  # <- progress bar





def animate_skeletons_with_video_fast(csv_file,
                                      tracks,
                                      save_path=None,
                                      fps=30,
                                      title="Skeleton Animation",
                                      show_progress=True):
    avi_fname = csv_file.replace(".csv", ".avi")
    cap = cv2.VideoCapture(avi_fname)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Could not open video file: {avi_fname}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_animals, n_frames, n_features, _ = tracks.shape
    n_frames = min(n_frames, total_frames)

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.axis("off")

    frame_display = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    colors = ["tab:blue", "tab:red", "tab:green"]

    # Pre-create one line per animal (7 keypoints connected)
    lines = [ax.plot([], [], "o-", lw=1.5, color=colors[i])[0] for i in range(n_animals)]

    if show_progress:
        pbar = tqdm(total=n_frames, desc="Rendering frames", ncols=80)

    def update(frame_idx):
        ret, frame = cap.read()
        if not ret:
            return lines

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.set_data(frame_rgb)

        for i, line in enumerate(lines):
            pts = tracks[i, frame_idx, :, :2]
            valid = ~np.isnan(pts[:, 0])
            line.set_data(pts[valid, 0], pts[valid, 1])

        if show_progress and frame_idx % 10 == 0:
            pbar.update(10)

        return [frame_display] + lines

    ani = FuncAnimation(fig,
                        update,
                        frames=n_frames,
                        interval=1000 / fps,
                        blit=True,       # ✅ huge speedup
                        repeat=False)

    if save_path:
        ani.save(save_path, writer="ffmpeg", fps=fps)
    else:
        plt.show()

    if show_progress:
        pbar.close()

    cap.release()
    plt.close(fig)
    return ani

#
def animate_skeletons_original(csv_file,
                               tracks,
                               save_path=None,
                               fps=30,
                               title="Raw YOLO Detections",
                               show_progress=True):
    """
    Animate all raw YOLO detections (no track matching) over the video background.

    Parameters
    ----------
    csv_file : str
        Path to the YOLO CSV file (used to infer .avi filename)
    tracks : np.ndarray
        Array of shape (n_tracks, n_frames, n_features, 3)
    save_path : str, optional
        If provided, saves animation to this path (e.g. .mp4 or .gif)
    fps : int, optional
        Frames per second (default 30)
    title : str, optional
        Figure title
    show_progress : bool, optional
        Show tqdm progress bar (default True)
    """
    # --- Infer video filename ---
    avi_fname = csv_file.replace(".csv", ".avi")
    cap = cv2.VideoCapture(avi_fname)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Could not open video file: {avi_fname}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"🎥 Loaded video: {avi_fname} ({width}×{height}, {total_frames} frames)")

    n_tracks, n_frames, n_features, _ = tracks.shape
    n_frames = min(n_frames, total_frames)

    # --- Set up figure ---
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.axis("off")
    ax.set_title(title)

    # --- Background video frame ---
    frame_display = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))

    # --- Scatter for all detections ---
    sc = ax.scatter([], [], s=8, color="tab:gray", alpha=0.7)
    text = ax.text(10, 20, "", fontsize=9, color="white", backgroundcolor="black")

    # --- Progress bar ---
    if show_progress:
        pbar = tqdm(total=n_frames, desc="Rendering frames", ncols=80)

    # --- Frame update function ---
    def update(frame_idx):
        ret, frame = cap.read()
        if not ret:
            return [frame_display, sc]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.set_data(frame_rgb)

        # Flatten all keypoints across all tracks
        frame_points = tracks[:, frame_idx, :, :2].reshape(-1, 2)
        valid = ~np.isnan(frame_points[:, 0])
        sc.set_offsets(frame_points[valid])

        text.set_text(f"Frame {frame_idx+1}/{n_frames}")

        if show_progress and frame_idx % 10 == 0:
            pbar.update(10)
        return [frame_display, sc, text]

    # --- Build animation ---
    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    # --- Save or show ---
    if save_path:
        print(f"💾 Saving animation to {save_path} ...")
        ani.save(save_path, writer="ffmpeg", fps=fps)
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()

    if show_progress:
        pbar.close()

    cap.release()
    plt.close(fig)
    return ani



def animate_skeletons_with_video(csv_file,
                                 tracks,
                                 save_path=None,
                                 fps=30,
                                 title="Skeleton Animation",
                                 show_progress=True):
    """
    Animate YOLO-tracked skeletons over the corresponding video frames.

    Parameters
    ----------
    csv_file : str
        Path to the YOLO CSV file (used to infer .avi filename)
    tracks : np.ndarray
        Array of shape (n_animals, n_frames, n_features, 3)
    save_path : str, optional
        If provided, saves animation as MP4 or GIF.
    fps : int, optional
        Frames per second (default = 30)
    title : str, optional
        Title of the animation.
    show_progress : bool, optional
        Whether to display tqdm progress bar.
    """
    avi_fname = csv_file.replace(".csv", ".avi")

    # --- Load video ---
    cap = cv2.VideoCapture(avi_fname)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Could not open video file: {avi_fname}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"🎬 Loaded video: {avi_fname} ({width}×{height}, {total_frames} frames)")

    # --- Tracking data info ---
    n_animals, n_frames, n_features, _ = tracks.shape
    n_frames = min(n_frames, total_frames)

    # --- Setup Matplotlib figure ---
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_title(title)
    ax.axis("off")

    colors = ["tab:blue", "tab:red", "tab:green"]
    scatters = [ax.scatter([], [], s=20, color=colors[i]) for i in range(n_animals)]

    # Background image (for video frames)
    frame_display = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))

    # --- Progress bar ---
    if show_progress:
        pbar = tqdm(total=n_frames, desc="Rendering frames", ncols=80)

    # --- Update function per frame ---
    def update(frame_idx):
        ret, frame = cap.read()
        if not ret:
            return scatters

        # Convert BGR (OpenCV) to RGB (Matplotlib)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.set_data(frame_rgb)

        for i, sc in enumerate(scatters):
            frame_data = tracks[i, frame_idx]
            valid = ~np.isnan(frame_data[:, 0])
            sc.set_offsets(frame_data[valid, :2])

        if show_progress and frame_idx % 10 == 0:
            pbar.update(10)

        return [frame_display] + scatters

    # --- Build animation ---
    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    # --- Save or display ---
    if save_path:
        print(f"Saving animation to {save_path} ...")
        ani.save(save_path, writer="ffmpeg", fps=fps)
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()

    if show_progress:
        pbar.close()

    cap.release()
    plt.close(fig)
    return ani






#
def plot_trajectory2(fname_csv,
                     centroid_orig, 
                     centroid_fixed, 
                     title=None):
    """
    Plot trajectories for 3 animals using blue, red, and green color shades.
    Alpha fades from light (early) to dark (late).
    Left: Original
    Right: Matched

    Parameters
    ----------
    centroid_orig : np.ndarray
        (3, n_frames, 2)
    centroid_fixed : np.ndarray
        (3, n_frames, 2)
    title : str, optional
        Figure title
    """
    # Define 3 base colors (fixed for all plots)
    base_colors = ["tab:blue", "tab:red", "tab:green"]
    n_animals = min(3, centroid_orig.shape[0])
    n_frames = centroid_orig.shape[1]

    # Generate fading alpha weights (light → dark)
    alphas = np.linspace(0.2, 1.0, n_frames)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    titles = ["Original Trajectories", "Matched Trajectories"]

    for j, (data, ax) in enumerate(zip([centroid_orig, centroid_fixed], axs)):
        for i in range(n_animals):
            centroid = data[i]
            valid = ~np.isnan(centroid[:, 0])

            # color array: same base RGB, varying alpha
            rgba = np.zeros((np.sum(valid), 4))
            rgb = plt.get_cmap()(0)  # dummy to fetch colors
            base_rgb = plt.matplotlib.colors.to_rgba(base_colors[i])
            rgba[:, :3] = base_rgb[:3]
            rgba[:, 3] = alphas[valid]

            sc = ax.scatter(
                centroid[valid, 0],
                centroid[valid, 1],
                color=rgba,
                s=15,
                label=f"Animal {i+1}",
            )

            # Connect with a thin line for readability
            ax.plot(
                centroid[valid, 0],
                centroid[valid, 1],
                lw=1.2,
                alpha=0.5,
                color=base_colors[i],
            )

        ax.set_title(titles[j])
        ax.set_xlabel("X position (px)")
        ax.set_ylabel("Y position (px)")
        ax.invert_yaxis()
        ax.set_xlim(0, 680)
        ax.set_ylim(0, 230)
        ax.legend()

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    # lets' just save od 

    plt.tight_layout()

    # lets save to disk instead of showing
    fname_out = fname_csv.replace('.csv', '.png')
    fig.savefig(fname_out, dpi=150)

    plt.close(fig)


# ------------------------------
# 4️⃣  Plotting utilities
# ------------------------------
def plot_trajectory(animal_id, ax, centroid, title=None, cmap="viridis"):
    """
    Plot XY trajectory of one animal, colored by time.
    """
    frames = np.arange(len(centroid))
    valid = ~np.isnan(centroid[:, 0])

    sc = ax.scatter(centroid[valid, 0],
                    centroid[valid, 1],
                    c=frames[valid],
                    cmap=cmap,
                    s=20,
                    alpha=0.9,
                    label=f"Animal {animal_id}")
    ax.plot(centroid[valid, 0],
            centroid[valid, 1],
            alpha=0.4, lw=1, color="gray")

    ax.set_xlabel("X position (px)")
    ax.set_ylabel("Y position (px)")
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    ax.legend()

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Frame index (time)")
    ax.set_ylim(0, 230)
    ax.set_xlim(0, 680)


#
def load_times(fname,
               minute):
    data = np.load(fname, allow_pickle=True)
    #print (data.files)
    frame_times_ms = data['frame_times']//1000
    #print (frame_times_ms)
    recording_start_time = data['recording_start_time']
    #print ("recording start time: ", recording_start_time)
    encoder_start = data['encoder_start']
    #print ("encoder start: ", encoder_start)

    ###############################################
    # convert july 24, 2025  exacdtly midngith to milisecond in epoch systm eimte clock but make sure its' UTC+1 london time
    #epoch_start = np.datetime64('2025-07-24T00:00:00', 'ms') - np.timedelta64(1, 'h')  # UTC+1
    #epoch_start_ms = epoch_start.astype('datetime64[ms]').astype(int)//1000

    # so this is the video time stamps relative to the epoch start
    #delta_times = frame_times_ms - epoch_start_ms
    #print("time srelative to midnight (in ms): ", delta_times)

    # now convert into bucket discrete time
    time_stamps_binned = frame_times_ms // 10 * 10 # convert to millseconds and round to 10ms bin
    #print("time relative to midnight in 10ms buckets: ", delta_times_bucket)

    # and convert into a discrete bin of 10ms from midnight
    #delta_times_bucket_discrete = delta_times_bucket // 10
    #print("time relative to midnight in 10ms discrete buckets: ", delta_times_bucket_discrete)

    return time_stamps_binned


# find the filename for each camera that falls in minute #1
def process_cams(cam_ids,
                 n_cores,
                 root_dir,
                 ram_disk_dir,
                 cage_id,
                 date,
                 hour_start,
                 minute,
                 parallel_flag=True,
                 shrink_factor=1,
                 skip_regeneration=False):

    if parallel_flag==False:
        for cam in cam_ids:
            print ("... processing camera: ", cam, " for minute: ", minute)
            decompress_cams(cam,
                            root_dir,
                            ram_disk_dir,
                            cage_id,
                            date,
                            hour_start,
                            minute,
                            shrink_factor=shrink_factor,
                            skip_regeneration=skip_regeneration)
    else:
        parmap.map(decompress_cams, 
                   cam_ids,
                    root_dir,
                    ram_disk_dir,
                    cage_id,
                    date,
                    hour_start,
                    minute,
                    shrink_factor=shrink_factor,
                    skip_regeneration=False,
                    pm_pbar=True,
                    pm_processes=n_cores,  # adjust based on your system)
        )

def get_ramdisk_dir(subdir="ramdisk"):
    """
    Returns a path to a RAM-backed directory (uses /dev/shm).
    Creates it if it doesn't exist.
    Works without sudo and is automatically cleared on reboot.
    """
    path = os.path.join("/dev/shm", subdir)
    os.makedirs(path, exist_ok=True)
    return path



# 
def decompress_cams(cam,
                    root_dir,
                    ram_disk_dir,
                    cage_id,
                    date,
                    hour_start,
                    minute,
                    shrink_factor=1,
                    skip_regeneration=False,
                    verbose=False      

                    ):

    #################################################
    ############## Generate Metadata ################
    #################################################
    
    fname_root = os.path.join(root_dir, 
                              str(cam),
                              date,
                              str(cage_id) + "_" +
                              str(cam) + "_" +
                              date+"_"+str(hour_start)+"_"+str(minute).zfill(2)+"*.npz")

    #print ("fname root: ", fname_root)
    # find any filenmes that match this pattern
    # There can only be 0 or 1 videos at most as we save 1 minute video per camera
    fnames = glob.glob(fname_root)

    #######################################################
    # if the camera has files we then need to load the frame times
    #######################################################
    if len(fnames)==0:
        if verbose:
            print ('... no files found for camera: ', cam, " minute: ", minute)
            print ("looking for files matching: ", fname_root)
            print ('')

        # --- FIX STARTS HERE ---
        # If no video for the current minute, check if there’s a temp file for this minute
        temp_path = os.path.join(
            ram_disk_dir,
            f"{cam}_shrink_{shrink_factor}_hour_{hour_start}_minute_{minute}_temp.bin"
        )
        clean_path = os.path.join(
            ram_disk_dir,
            f"{cam}_shrink_{shrink_factor}_hour_{hour_start}_minute_{minute}_clean.bin"
        )

        if os.path.exists(temp_path):
            shutil.move(temp_path, clean_path)

        # we also need to pad this file so it has 6000 frames
        if os.path.exists(clean_path):
            if verbose:
                print ("... padding existing clean file to full 6000 frames: ", clean_path)
            frame_height = 720 // shrink_factor
            frame_width = 1280 // shrink_factor
            channels = 3
            frame_size_bytes = frame_height * frame_width * channels
            total_frames = 6000
            current_size = os.path.getsize(clean_path)
            current_frames = current_size // frame_size_bytes
            frames_to_add = total_frames - current_frames

            if frames_to_add > 0:
                blank_frame = np.zeros((frame_height, frame_width, channels), dtype=np.uint8)
                with open(clean_path, 'ab') as f:
                    for _ in range(frames_to_add):
                        f.write(blank_frame.tobytes())
                if verbose:
                    print (f"... added {frames_to_add} blank frames to {clean_path}")

        return 

    #######################################################
    ############### CONVERT TIME STAMPS ####################
    #######################################################
    if verbose:
        print ("minute:", minute, " cam:", cam ) #, " files:", fnames)

    # load the time stamps for this camera
    time_stamps_binned = load_times(fnames[0], minute)
    if verbose:
        print ("time stamps binned: ", time_stamps_binned)

    # Convert to UTC+1 using timezone offset
    dt_naive = datetime.strptime(f"{date.replace('_', '-')} {hour_start}:{minute:02d}", "%Y-%m-%d %H:%M")
    dt_utc1 = dt_naive.replace(tzinfo=timezone(timedelta(hours=1)))  # UTC+1

    # Get absolute time in nanoseconds
    timestamp_ns = int(dt_utc1.timestamp() * 1_000_000_000)

    # Truncate to milliseconds
    unix_time_to_start_of_minute = timestamp_ns // 1_000_000
    if verbose:
        print("absolute unix time (ms, UTC+1) to start of minute:", unix_time_to_start_of_minute)

    #################################################
    ############### Decompress video ################
    #################################################
    # process #1
    fname_video = fnames[0].replace('_metadata.npz', '.h264') 
    if os.path.exists(fname_video)==False:
        fname_video = fnames[0].replace('_metadata.npz', '.mp4') 

    # move the fname_video file to ramdrive
    fname_video_ramdisk = os.path.join(ram_disk_dir, os.path.basename(fname_video))
    if not os.path.exists(fname_video_ramdisk):
        if verbose:
            print ("... copying video file to ramdisk: ", fname_video_ramdisk)
        shutil.copy(fname_video, fname_video_ramdisk)

    # use opencv to uncomrpess the video to .png files on disk
    decompress_video(minute,
                     hour_start,
                    fname_video_ramdisk,
                    root_dir,
                    cam,
                    time_stamps_binned,
                    unix_time_to_start_of_minute, 
                    shrink_factor,
                    skip_regeneration=skip_regeneration,
                    overwrite_existing=True)     
    print ('')


#
def decompress_video(minute, 
                     hour_start,
                     fname,
                     root_dir,
                     cam,
                     time_stamps_binned,
                     unix_time_to_start_of_minute,
                     shrink_factor=1,
                     skip_regeneration=False,
                     overwrite_existing=False,
                     verbose=False):
    
    ''' use OpenCV or something fast to load the video file or uncompress it.
        Not clear yet if we should just save to disk, but probably,
    '''

    #verbose = True
    # use opencv to load video
    # and save the png files with the filenmae of 
    
    # fixed movie parameters
    frame_height = 720
    frame_width  = 1280
    channels = 3

    # make file names. 
    # TODO: The goal is to always work on a _temp.bin file until we clean exit then we
    # either keep it as _temp.bin (for next minute), or rename it to .bin for current minute.
    # 1. For the current minute we need to check if there's a _temp.bin files which means it 
    # was already started previously.
    # 2. If it exists, then we make a copy as .bin and append to it. 
    # 2.1 after finishing appending to it we rename it to .bin and delete the _temp.bin 
    fname_video_current_minute_clean= os.path.split(fname)[0]+"/"+str(cam) + "_shrink_"+ str(shrink_factor) \
                    + "_hour_" + hour_start + "_minute_" + str(minute) + "_clean.bin"
    fname_video_current_minute_temp = os.path.split(fname)[0]+"/"+str(cam) + "_shrink_"+ str(shrink_factor) \
                    + "_hour_" + hour_start + "_minute_" + str(minute) + "_temp.bin"
    fname_video_next_minute_temp = os.path.split(fname)[0]+"/"+str(cam) + "_shrink_"+ str(shrink_factor) \
                    + "_hour_" + hour_start + "_minute_" + str(minute+1) + "_temp.bin"

    # check if the clean file already exists
    if skip_regeneration and os.path.exists(fname_video_current_minute_clean):
        if verbose:
            print ("... skipping regeneration of video file: ", fname_video_current_minute_clean)
        return

    # shrink based on the shrink factor
    if shrink_factor > 1:
        frame_height //= shrink_factor
        frame_width  //= shrink_factor
        if verbose:
            print ("shrinking video frames to: ", frame_height, "x", frame_width)
    
    # make blank frame 
    frame_size_bytes = frame_height * frame_width * channels
    blank_frame = np.zeros((frame_height, frame_width, channels), dtype=np.uint8)

    # we offset the loaded time steps to the start of the current minute
    # so we can then use a 6000 bin fixed size video structure 
    times_relative_to_min_start = time_stamps_binned - unix_time_to_start_of_minute
    if verbose:
        print ("time stamps relative to start of minute: ", times_relative_to_min_start)
        print (" total frames: ", len(times_relative_to_min_start), 
             "total unique frames:", np.unique(times_relative_to_min_start).shape[0])

    # Step 1: Check if file exists & count existing frames
    # then we need to indepnt to existin frame to start appending
    # TODO: the problem here is if the code crashes, then it appends incorrect # of frames so we'
    # will never be able to correct it;
    # SOLUTION: seems to be to make a temporary .bin file and then only finalize it when we
    # have cleanly written all frames
    
    # So, first check if there was a temp file saved from the previuos minute
    # this means that we have a good video file to start with - but haven't completed the current minute
    if os.path.exists(fname_video_current_minute_temp):
        if verbose:
            print ("... found temp video file for current minute: ", 
                   fname_video_current_minute_clean)
        file_size = os.path.getsize(fname_video_current_minute_temp)
        frames_already_written = file_size // frame_size_bytes
        if verbose:
            print(f"File already exists: {frames_already_written} frames found.")

        # let's make a copy of the temp file to the current minute
        # we can now append to the fname_current_minute as required - and 
        #shutil.copy(fname_video_current_minute_clean, fname_video_current_minute_temp)

    else:
        frames_already_written = 0
        if verbose:
            print("File does not exist yet.")

    # need to check that if there is a full video in place already then we can exit safely
    if frames_already_written >= 6000:
        if verbose:
            print ("... this minute of data already fully decompressed, exiting ...")
            print (" --- THIS CASE SHOULDNT HAPPNE... to check...")
        return

    # we need to make sure that the first time stamp is 0
    # so we now loop over the video and the raw frames 
    # we fill blanks until we reach the next time stamp

    # use open cv to laod video
    import cv2
    if verbose:
        print ("opening video file for reading: ", fname)
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        raise IOError("Cannot open video file: {}".format(fname))

    #
    ctr_frame = 0
    ctr_bin = frames_already_written*10
    n_frames_read = 0
    n_unique_frames_written = 0

    # ok so we now work only on the current_minute_temp
    # until we have a clean exit - in which case we overwrite the clean file
    with open(fname_video_current_minute_temp, 'ab') as f:
        
        ##################### FILL EXISTING VIDEO WITH BLANKS ######################
        # if video already in place need to then fill the gap between the previous video ended
        #  and current one; usually about 3-5 seconds of duration during file upload to the server
        if frames_already_written > 0:
            if verbose:
                print ("...  filling up preexisting video with blank frames #: ", frames_already_written)
            # TODO: fill in video with last frame - so we load it
            while ctr_bin != times_relative_to_min_start[ctr_frame]:
                # we write on frame to stack and advance 10 ms
                f.write(blank_frame.tobytes())
                ctr_bin += 10
                # print ("ctr_bin: ",ctr_bin)
                # print ("ctr_frame: ", ctr_frame)
                # print ("times_relative to min start: ", times_relative_to_min_start[ctr_frame])
                # don't increment the video frame coutner; 
                # we're just trying to catch up to it with the blank frames

        ####################################################################
        ##################### LOOP OVER 60 sec VIDEO  ######################
        ####################################################################
        while ctr_bin < 60000:
            ret, frame = cap.read()
            if not ret:
                break

            # let's print size of the frame
            #print ("frame shape: ", frame.shape, " frame size: ", frame.size, " bytes: ", frame.nbytes)

            #return

            #if ctr_bin%5000==0:
            #    print ("processing frame: ", ctr_bin, " / ", 60000, " frames written: ", n_frames_written)

            # we subsample/shrink frame based on shrink factor
            if shrink_factor > 1:
                # let's use simple subsampling
                if True:
                    frame = frame[::shrink_factor, ::shrink_factor, :]
                else:
                    #
                    frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

            # 
            while ctr_bin < times_relative_to_min_start[ctr_frame]:
                # we need to write a blank frame
                f.write(blank_frame.tobytes())
                ctr_bin += 10  # increment the bin counter
                # increment the bin counter

            # write the frame to the binary file
            #print ("ctr_bin", ctr_bin, ", ctr_frame: ", ctr_frame, times_relative_to_min_start[ctr_frame])
            n_frames_read += 1

            # check to makes sure next value is different:
            # we only write frames that are unique
            # don't generallyneed to check if we have more ctr_frame than 
            if (ctr_frame+1)<=times_relative_to_min_start.shape[0]:
                if times_relative_to_min_start[ctr_frame]!=times_relative_to_min_start[ctr_frame+1]:
                    f.write(frame.tobytes())
                    ctr_bin += 10
                    n_unique_frames_written += 1

            #
            ctr_frame += 1

            # we also replace the blank frame now with the last read frame
            blank_frame = frame.copy()

    # TODO: Need an extra step to indicate clean exit;
    # so we'll need to rename the _temp file to _clean.bin or something like this
    if verbose:
        print ("... clean exit, renaming current minute temp file to clean file ...")
    shutil.move(fname_video_current_minute_temp, fname_video_current_minute_clean)
    
    if verbose:
        print ("# UNIQUE frames written current min", n_unique_frames_written,
            ", n_frames_read: ", n_frames_read)
        print ("last frame time written: ", times_relative_to_min_start[ctr_frame], " ctr_frame: ", ctr_frame)

    #########################################################################
    ################# Extact frames from 2nd minute #########################
    #########################################################################
    # let's clip the frames relative to start of 2nd miunte so basically ctr_frame

    # check if there are aneough frame sleft
    if times_relative_to_min_start[ctr_frame:].shape[0]==0:
        return

    times_relative_to_min_start = times_relative_to_min_start[ctr_frame:] - 60000 +10
    if verbose:
        print ("writing second minute frames starting at: ", times_relative_to_min_start[:5])

       # write the rest of the frames to the next vid
    ctr_bin = 0 
    n_frames_read = 0
    n_unique_frames_written = 0
    ctr_frame = 0

    # Step 2: Open in append mode ('ab' = append binary)
    # we only ever write to a _temp file here; it will be converted to a clean file on the next cycle
    with open(fname_video_next_minute_temp, 'wb') as f:

        #
        while ctr_bin < 60000:
            ret, frame = cap.read()
            if not ret:
                #print ("... NO MORE VID FRAMES...")
                break
            #
            #if ctr_bin%5000==0:
            #    print ("processing frame: ", ctr_bin, " / ", 60000, " frames written: ", n_frames_written)

            # apply shrink factor
            if shrink_factor > 1:
                # let's use simple subsampling
                if True:
                    frame = frame[::shrink_factor, ::shrink_factor, :]
                else:
                    #
                    frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

            # 
            #print ("ctr_bin", ctr_bin, ", ctr_frame: ", ctr_frame, times_relative_to_min_start[ctr_frame])
            while ctr_bin != times_relative_to_min_start[ctr_frame]:
                # we need to write a blank frame
                f.write(blank_frame.tobytes())
                ctr_bin += 10  # increment the bin counter
                # increment the bin counter

            #
            if (ctr_frame+1)<=(times_relative_to_min_start.shape[0]-1):
                if times_relative_to_min_start[ctr_frame]!=times_relative_to_min_start[ctr_frame+1]:
                    f.write(frame.tobytes())
                    ctr_bin += 10
                    n_unique_frames_written += 1

            # write the frame to the binary file
            n_frames_read += 1

            # check to makes sure next value is different:
            if ctr_frame+1 >= len(times_relative_to_min_start):
                #print ("... NO MORE TIME STAMPS...")
                break


            #
            ctr_frame += 1
            
            # we also replace the blank frame now with the last read frame
            blank_frame = frame.copy()

    if verbose:
        print ("# UNIQUE frames written to next min video", n_unique_frames_written,
              ", n_frames_read: ", n_frames_read)
        
    # delete the .h264 file
    os.remove(fname)



# function to get width, height from crop
def get_size(crop):
    x0, x1, y0, y1 = crop
    return x1 - x0, y1 - y0

def get_video_size(nrows, 
                    ncols,
                    rows,
                    crops):
    
    ''' get the final video size based on the crop table and the rows/cols of cameras '''
    # compute row widths and col heights
    row_heights = []
    for r in range(nrows):
        heights = []
        for c in range(ncols):
            cam_id = rows[r, c]
            crop = crops[f"cam{cam_id}"]
            w, h = get_size(crop)
            heights.append(h)
        row_heights.append(max(heights))  # height is uniform per row

    col_widths = []
    for c in range(ncols):
        widths = []
        for r in range(nrows):
            cam_id = rows[r, c]
            crop = crops[f"cam{cam_id}"]
            w, h = get_size(crop)
            widths.append(w)
        col_widths.append(max(widths))  # width is uniform per column

    # final frame size
    final_width = sum(col_widths)
    final_height = sum(row_heights)

    print(f"Final video size: {final_width} x {final_height}")
    return final_width, final_height


import os

def delete_bins(ram_disk_dir, 
                n_cams, 
                minute, 
                shrink_factor=1, 
                verbose=False):
    """
    Delete the known *_clean.bin files for all cameras after video creation.

    Parameters
    ----------
    ram_disk_dir : str
        Path to the RAM disk directory (e.g. /dev/shm/ramdisk).
    n_cams : int
        Number of camera channels used.
    minute : int
        The minute index used in file naming.
    shrink_factor : int, optional
        Shrink factor used in file naming (default = 1).
    verbose : bool
        Print deletion progress.

    Returns
    -------
    int : Number of files successfully deleted.
    """

    # also delete all .h264 files in the ramdisk dir
    h264_files = glob.glob(os.path.join(ram_disk_dir, "*.h264"))
    for h264_file in h264_files:
        os.remove(h264_file)
        
    # also delete all .h264 files in the ramdisk dir
    mp4_files = glob.glob(os.path.join(ram_disk_dir, "*.mp4"))
    for mp4_file in mp4_files:
        os.remove(mp4_file)

    # delte all _clean.bin files
    clean_files = glob.glob(os.path.join(ram_disk_dir, "*clean.bin"))
    for clean_file in clean_files:
        os.remove(clean_file)

    # delete previous _temp.bin files
    temp_files = glob.glob(os.path.join(ram_disk_dir, "*"+str(minute-1)+ "_temp.bin"))
    for temp_file in temp_files:
        os.remove(temp_file)


#
def make_video(root_dir,
               ram_disk_dir,
               date,
                minute,
                hour_start,
                n_cams,
                fname_combined,
                shrink_factor=1,
                frame_subsample=1,
                overwrite_existing = False,
                delete_bins_flag = False,
                ):
    
    from tqdm import trange
    ''' make a video from the available frames for this minute '''

    rows = np.array([
        [16, 13, 10, 7, 4, 1],
        [17, 14, 11, 8, 5, 2],
        [18, 15, 12, 9, 6, 3]
    ])
    nrows, ncols = 3, 6

    # load translation table
    fname_crop_table ="crop_table.yaml"
    crops = yaml.safe_load(open(fname_crop_table, 'r'))
    #print ("crop table: ", crops)
    #crop table:  {'cam1': [150, 1280, 0, 720], 'cam2': [150, 1280, 0, 720], 'cam3': [150, 1280, 0, 720], 'cam4': [100, 1180, 0, 720], 'cam5': [100, 1180, 0, 720], 'cam6': [100, 1180, 0, 720], 'cam7': [100, 1180, 0, 720], 'cam8': [100, 1180, 0, 720], 'cam9': [100, 1180, 0, 720], 'cam10': [100, 1180, 0, 720], 'cam11': [100, 1180, 0, 720], 'cam12': [100, 1180, 0, 720], 'cam13': [100, 1180, 0, 720], 'cam14': [100, 1180, 0, 720], 'cam15': [100, 1180, 0, 720], 'cam16': [0, 1180, 0, 720], 'cam17': [0, 1180, 0, 720], 'cam18': [0, 1180, 0, 720]}

    # ok need to figure out the fully frame size based on these crops and the rows that they will be appended above

    vid_width, vid_height = get_video_size(nrows, 
                                                ncols,
                                                rows,
                                                crops)
    
    if (shrink_factor > 1):
        vid_width //= shrink_factor
        vid_height //= shrink_factor
        #print ("shrinking final video to: ", vid_width, "x", vid_height)

    #
    #################################################################
    #    
    frame_height = 720        # pixels
    frame_width = 1280       # pixels
    channels = 3               # RGB
    if shrink_factor > 1:
        frame_height //= shrink_factor
        frame_width  //= shrink_factor
        #print ("shrinking video frames to: ", frame_height, "x", frame_width)

    # make a blank default image
    frame_blank = np.zeros((frame_height, frame_width, channels), dtype=np.uint8)

    #
    frame_size_bytes = frame_height * frame_width * channels  # 1280 * 768 * 3 = 2,359,296 bytes

    if os.path.exists(fname_combined) and overwrite_existing==False:
        #print ("... combined video file already exists: ", fname_combined)
        return

    #frame_all_cams_blank = np.zeros((frame_height * rows, frame_width * cols, channels), dtype=np.uint8)
    #
    #print ("creating combined video file: ", fname_combined)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(fname_combined, 
                          fourcc, 
                          30.0, 
                          #
                          (vid_width, vid_height)
                          #(frame_width * ncols, frame_height * nrows)
                          )

    # we actually want to grab freeze frames from non-recorded cameras - so we don't reinitialize this
    for i in trange(0,6000, frame_subsample):
        #
        frame_all_cams_rows = [[],[],[]]

    
        # loop over cameras and grab a frame from each
        for cam in range(n_cams,0,-1):
            # find the filename for this camera and bin
            fname_frame = os.path.join(ram_disk_dir,
                                    str(cam) + "_shrink_" + str(shrink_factor) +
                                    "_hour_" + hour_start + "_minute_"+
                                    str(minute) + "_clean.bin")
            
            #
            if os.path.exists(fname_frame):
                
                # we need to index into this file to the current frame i using framesize-bytes
                with open(fname_frame, 'rb') as f:
                    f.seek(i * frame_size_bytes)

                    frame_data = f.read(frame_size_bytes)
                    if len(frame_data) != frame_size_bytes:
                        # leave the previous frame in place
                        continue
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((frame_height, frame_width, channels))

                    try:
                        frame = np.frombuffer(frame, dtype=np.uint8).reshape((frame_height, 
                                                                          frame_width, 
                                                                          channels))
                    except:
                        # ok we need to print a lot of metadata to understan why we're reading patst the end of the file
                        #pass
                        frame = frame_blank.copy()
                        # print ("i: ", i)
                        # print ("cam: ", cam)
                        # print ("fname: ", fname_frame)
                        # print ("frame_size_bytes: ", frame_size_bytes)
            else:
                frame = frame_blank.copy()

            #
            #print ("frame shape: ", frame.shape)
            # find row of the camera
            result = np.where(rows == cam)
            #print ("result", result)
            row_index = result[0][0] 
            #print ("row: ", row_index)

            # we want to save each of these frames as png for offling analysis
            if False:
            #if i in frame_ids_align:
                fname_out = os.path.join(root_dir, "frames",
                                        "align_frame_" + str(minute).zfill(2) + "_" + str(i).zfill(4) +
                                        "_cam" + str(cam) + ".png")
                cv2.imwrite(fname_out, frame)

            # let's now index into the frame based on the crop-table
            # first grab the camera specific crop
            # this is how the data is stored: cam1: [100,1280,0,720]
            crop = crops[f"cam{cam}"]
            x1, x2, y1, y2 = crop
            # shrink the coords by the required amountd
            x1 //= shrink_factor
            x2 //= shrink_factor
            y1 //= shrink_factor
            y2 //= shrink_factor
            #print ('frame', frame.shape, " crop: ", x1, y1, x2, y2)
            
            # if y1 is > 0 then we need to just
            if y1>0:
                # we need to grab the frame and do a roll and fill with blanks
                frame = np.roll(frame, y1, axis=0)
                frame[:y1,:,:] = 0
                frame = frame[:, x1:x2]
            else:
                frame = frame[y1:y2, x1:x2]
            #print ("final frame shape: ", frame.shape)
            # 
        
            #
            frame_all_cams_rows[row_index].append(frame)


        # now hstack within each row
        row_images = []
        for k in range(nrows):
            row_images.append(np.hstack(frame_all_cams_rows[k]))

        # then vstack across rows
        try:
            frame_all_cams_blank = np.vstack(row_images)
        except:
            for k in range(len(row_images)):
                print ("row image shape: ", row_images[k].shape)
            print ("... error vstacking row images ...")

        #return
        # print ("frame all cams shape: ", frame_all_cams_blank.shape)

        #return

        #print ("frame all cams shape: ", frame_all_cams_blank.shape)
        #return
        
        # now write the combined frame to the video file
        out.write(frame_all_cams_blank)

        # also save the img to disk
        if False:
            fname = os.path.join(root_dir, "frames", "frame_" + str(minute).zfill(2) + "_" + str(i).zfill(4) + ".png")
            cv2.imwrite(fname, frame_all_cams_blank)

    # this is probably coming at the end. not sure exactly
    out.release()
    #print ('finished writing combined video file: ', fname_combined)

    # delete the bin files
    if delete_bins_flag:
        delete_bins(ram_disk_dir, 
                    n_cams, 
                    minute, 
                    shrink_factor=shrink_factor, 
                    verbose=True)

