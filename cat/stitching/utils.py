import numpy as np
import os
import subprocess
import glob
import cv2
import shutil  # <-- needed for copying
import yaml
import parmap
from datetime import datetime, timezone, timedelta

#
def load_times(
                fname,
                inter_frame_interval,
                minute):

    data = np.load(fname, allow_pickle=True)
    frame_times_ms = data['frame_times']//1000
    recording_start_time = data['recording_start_time']
    encoder_start = data['encoder_start']

    ###############################################
    # convert july 24, 2025  exacdtly midngith to milisecond in epoch systm eimte clock but make sure its' UTC+1 london time
    #epoch_start = np.datetime64('2025-07-24T00:00:00', 'ms') - np.timedelta64(1, 'h')  # UTC+1
    #epoch_start_ms = epoch_start.astype('datetime64[ms]').astype(int)//1000

    # so this is the video time stamps relative to the epoch start
    #delta_times = frame_times_ms - epoch_start_ms
    #print("time srelative to midnight (in ms): ", delta_times)

    # now convert into bucket discrete time
    time_stamps_binned = frame_times_ms // inter_frame_interval * inter_frame_interval # convert to millseconds and round to 10ms bin
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
    
    #
    print ("Processing cams...")

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
def get_ramdisk_dir_non_memory(subdir="ramdisk"):
    """
    Returns a path to a RAM-backed directory (uses /dev/shm).
    Creates it if it doesn't exist.
    Works without sudo and is automatically cleared on reboot.
    """
    path = os.path.join("/home/cat/ramdisk", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def file_not_found(fname_root,
                    cam,
                    minute,
                    hour_start,
                    shrink_factor,
                    ram_disk_dir,
                    resolution,
                    verbose=False
                ):
    
    #
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
        frame_height = resolution[1] // shrink_factor
        frame_width = resolution[0] // shrink_factor
        channels = 3
        frame_size_bytes = frame_height * frame_width * channels
        total_frames = frame_rate * duration  # 6000 frames for 1 minute at 100 fps
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
    
    #
    verbose = True

    # load meta data from config file
    fname_config = os.path.join(root_dir, "config.yaml")
    config = yaml.safe_load(open(fname_config, 'r'))

    resolution = config['resolution']
    print ("resolution: ", resolution)

    frame_rate = config['frame_rate']
    print ("frame rate: ", frame_rate)

    inter_frame_interval = 1000 // frame_rate  # in milliseconds

    duration = config['duration']
    print ("duration: ", duration)


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
        file_not_found(fname_root,
                        cam,
                       minute,
                       hour_start,
                       shrink_factor,
                       ram_disk_dir,
                       resolution,
                       verbose=verbose)
        return


    #######################################################
    ############### CONVERT TIME STAMPS ####################
    #######################################################
    if verbose:
        print ("minute:", minute, " cam:", cam ) #, " files:", fnames)

    # load the time stamps for this camera
    time_stamps_binned = load_times(fnames[0],
                                    inter_frame_interval,
                                    minute)
    if verbose:
        print ("time stamps binned: ", time_stamps_binned)

    # Convert to UTC+1 using timezone offset
    dt_naive = datetime.strptime(f"{date.replace('_', '-')} {hour_start}:{minute:02d}", "%Y-%m-%d %H:%M")
    dt_utc1 = dt_naive.replace(tzinfo=timezone(timedelta(hours=0)))  # UTC+1

    # Get absolute time in nanoseconds
    timestamp_ns = int(dt_utc1.timestamp() * 1_000_000_000)

    # Truncate to milliseconds
    unix_time_to_start_of_minute = timestamp_ns // 1_000_000
    if verbose:
        print ("date: ", date, " hour_start: ", hour_start, " minute: ", minute)
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

    verbose = True
    # use opencv to load video
    # and save the png files with the filenmae of 
    
    # fixed movie parameters
    fname_config = os.path.join(root_dir, "config.yaml")
    config = yaml.safe_load(open(fname_config, 'r'))
    resolution = config['resolution']

    # 
    frame_rate = config['frame_rate']

    #
    duration = config['duration']

    print ("Decompressing video for cam: ", cam,
           ", minute: ", minute,
           ", fname: ", fname)

    # inter frame interval in milliseconds
    inter_frame_interval = 1000 // frame_rate  # in milliseconds


    #
    frame_height = resolution[1]
    frame_width  = resolution[0]
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
    # we want to get nerestine
    if shrink_factor > 1:
        frame_height = int(round(frame_height / shrink_factor))
        frame_width  = int(round(frame_width  / shrink_factor))
        if verbose:
            print ("shrinking video frames to: ", frame_height, "x", frame_width)
    
    # make blank frame 
    frame_size_bytes = frame_height * frame_width * channels
    blank_frame = np.zeros((frame_height, frame_width, channels), dtype=np.uint8)

    # helper util for verbose debugging of frame writes
    def log_frame_write(target_label,
                        frame_idx,
                        ctr_bin_val,
                        frame_kind,
                        frame_obj,
                        ts_target=None):
        if not verbose:
            return
        if frame_obj is None:
            shape = (frame_height, frame_width, channels)
            nbytes = frame_size_bytes
        else:
            shape = frame_obj.shape
            nbytes = frame_obj.nbytes
        msg = (f"[{target_label}] frame_idx={frame_idx} ctr_bin_ms={ctr_bin_val} "
               f"type={frame_kind} shape={shape} bytes={nbytes}")
        if ts_target is not None:
            msg += f" target_timestamp_ms={ts_target}"
        print(msg)

    # we offset the loaded time steps to the start of the current minute
    # so we can then use a 6000 bin fixed size video structure 
    times_relative_to_min_start = time_stamps_binned - unix_time_to_start_of_minute
    # 
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
        frames_already_written = file_size / frame_size_bytes
        if verbose:
            print(f"File already exists: {frames_already_written} frames found.")
        frames_already_written = int(frames_already_written)

        # let's make a copy of the temp file to the current minute
        # we can now append to the fname_current_minute as required - and 
        #shutil.copy(fname_video_current_minute_clean, fname_video_current_minute_temp)

    else:
        frames_already_written = 0
        if verbose:
            print("File does not exist yet.")

    # need to check that if there is a full video in place already then we can exit safely
    if frames_already_written >= (frame_rate * duration):
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
    # this should index by the inter frame interval in milliseconds not fixed 10 ms
    ctr_bin = frames_already_written*inter_frame_interval
    n_frames_read = 0
    n_unique_frames_written = 0

    # ok so we now work only on the current_minute_temp
    # until we have a clean exit - in which case we overwrite the clean file
    with open(fname_video_current_minute_temp, 'ab') as f:
        frames_written_clean = frames_already_written
        
        ##################### ADvance INDEX ######################
        # if video already advance index
        #  and current one; usually about 3-5 seconds of duration during file upload to the server
        # but we shoudln't overwrite the existing video with blank frames; just need to move the index forward
        # so we need to check if the current frame time is greater than the previous frame time
        if frames_already_written > 0:
            if verbose:
                print ("...  advancing video  frames #: ", frames_already_written)
            # TODO: fill in video with last frame - so we load it
            if ctr_frame < len(times_relative_to_min_start):
                while ctr_bin != times_relative_to_min_start[ctr_frame]:
                    # we write on frame to stack and advance 10 ms
                    log_frame_write("CURRENT_MINUTE_CLEAN",
                                    frames_written_clean,
                                    ctr_bin,
                                    "no-fill-advance-index",
                                    blank_frame,
                                    ts_target=times_relative_to_min_start[ctr_frame])
                    # i don't think we need to write the blank frame to the file; we just need to move the index forward
                    #f.write(blank_frame.tobytes())
                    frames_written_clean += 1
                    ctr_bin += inter_frame_interval
                    # print ("ctr_bin: ",ctr_bin)
                    # print ("ctr_frame: ", ctr_frame)
                    # print ("times_relative to min start: ", times_relative_to_min_start[ctr_frame])
                    # don't increment the video frame coutner; 
                    # we're just trying to catch up to it with the blank frames


        # this is required for the case where the recording starts mid-minute
        if ctr_frame < len(times_relative_to_min_start):
            while ctr_bin < times_relative_to_min_start[ctr_frame]:
                # we need to write a blank frames until we reach the first frame time in the current minute
                log_frame_write("CURRENT_MINUTE_CLEAN",
                                frames_written_clean,
                                ctr_bin,
                                "inserting blank frame to advance index",
                                blank_frame,
                                ts_target=times_relative_to_min_start[ctr_frame])
                f.write(blank_frame.tobytes())
                frames_written_clean += 1
                ctr_bin += inter_frame_interval  # increment the bin counter
                # increment the bin counter

        # placehodler in case there's an error on the first frame
        prev_frame = blank_frame.copy()

        ####################################################################
        ##################### LOOP OVER 60 sec VIDEO  ######################
        ####################################################################
        while ctr_bin < duration * 1000:  # 60000 ms = 60 seconds
        # lets use 
            ret, frame = cap.read()
            if not ret:
                break

            # we subsample/shrink frame based on shrink factor
            if shrink_factor > 1:
                # let's use simple subsampling
                if True:
                    frame = frame[::shrink_factor, ::shrink_factor, :]
                else:
                    #
                    frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)


            # this is required for the case where there is a frame time gap in the recording
            # we jsut write the previuos frame
            # so here we're comparing the current bin index with expected bin index for the current frame
            if ctr_frame < len(times_relative_to_min_start):
                while ctr_bin < times_relative_to_min_start[ctr_frame]:
                    # we need to write a blank frames until we reach the first frame time in the current minute
                    log_frame_write("CURRENT_MINUTE_CLEAN",
                                    frames_written_clean,
                                    ctr_bin,
                                    "mid_recording_duplicate_frame - advance index",
                                    prev_frame,
                                    ts_target=times_relative_to_min_start[ctr_frame])
                    f.write(blank_frame.tobytes())
                    frames_written_clean += 1
                    ctr_bin += inter_frame_interval  # increment the bin counter
                    # increment the bin counter
            else:
                # No more frames in times_relative_to_min_start, break out of loop
                break


            # write the frame to the binary file
            #print ("ctr_bin", ctr_bin, ", ctr_frame: ", ctr_frame, times_relative_to_min_start[ctr_frame])
            n_frames_read += 1

            # check to makes sure next value is different:
            # we only write frames that are unique
            # don't generallyneed to check if we have more ctr_frame than 
            if (ctr_frame+1) < times_relative_to_min_start.shape[0]:
                if times_relative_to_min_start[ctr_frame]!=times_relative_to_min_start[ctr_frame+1]:
                    log_frame_write("CURRENT_MINUTE_CLEAN",
                                    frames_written_clean,
                                    ctr_bin,
                                    "real_frame",
                                    frame,
                                    ts_target=times_relative_to_min_start[ctr_frame])
                    f.write(frame.tobytes())
                    frames_written_clean += 1
                    ctr_bin += inter_frame_interval
                    n_unique_frames_written += 1

            #
            ctr_frame += 1
            
            # Check if we've exhausted all frames in times_relative_to_min_start
            if ctr_frame >= len(times_relative_to_min_start):
                break

            # we also replace the blank frame now with the last read frame
            prev_frame = frame.copy()

            #
            #prev_frame = frame.copy()

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

    times_relative_to_min_start = times_relative_to_min_start[ctr_frame:] - duration*1000 + inter_frame_interval
    if verbose:
        print ("writing second minute frames starting at: ", times_relative_to_min_start[:5])

    #return 

       # write the rest of the frames to the next vid
    ctr_bin = 0 
    n_frames_read = 0
    n_unique_frames_written = 0
    ctr_frame = 0

    # Step 2: Open in append mode ('ab' = append binary)
    # we only ever write to a _temp file here; it will be converted to a clean file on the next cycle
    with open(fname_video_next_minute_temp, 'wb') as f:
        frames_written_next_temp = 0

        #
        while ctr_bin < duration*1000:
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
            if ctr_frame >= len(times_relative_to_min_start):
                break
            while ctr_bin != times_relative_to_min_start[ctr_frame]:
                # we need to write a blank frame
                log_frame_write("NEXT_MINUTE_TEMP",
                                frames_written_next_temp,
                                ctr_bin,
                                "blank_gap_fill",
                                blank_frame,
                                ts_target=times_relative_to_min_start[ctr_frame])
                f.write(blank_frame.tobytes())
                frames_written_next_temp += 1
                ctr_bin += inter_frame_interval  # increment the bin counter
                # increment the bin counter

            #
            if (ctr_frame+1) < times_relative_to_min_start.shape[0]:
                if times_relative_to_min_start[ctr_frame]!=times_relative_to_min_start[ctr_frame+1]:
                    log_frame_write("NEXT_MINUTE_TEMP",
                                    frames_written_next_temp,
                                    ctr_bin,
                                    "real_frame",
                                    frame,
                                    ts_target=times_relative_to_min_start[ctr_frame])
                    f.write(frame.tobytes())
                    frames_written_next_temp += 1
                    ctr_bin += inter_frame_interval
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

#
def get_video_size(nrows, 
                   ncols,
                   rows,
                   crops):

    print ("crop table: ", crops)
    print ("rows: ", rows)
    print ("ncols: ", ncols)
    print ("nrows: ", nrows)
    
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
def make_video(config,
               flip_vertical_flag,
               root_dir,
               ram_disk_dir,
               date,
                minute,
                hour_start,
                n_cams,
                cam_ids,
                fname_combined,
                shrink_factor=1,
                frame_subsample=1,
                overwrite_existing = False,
                delete_bins_flag = False,
                ):
    
    from tqdm import trange
    ''' make a video from the available frames for this minute '''

    n_cams = config['n_cams']

    #
    resolution = config['resolution']

    #
    frame_rate = config['frame_rate']
    duration = config['duration']
    cage_id = config['cage_id']


    # Cage 1 layout
    if n_cams == 18:
        rows = np.array([
            [16, 13, 10, 7, 4, 1],
            [17, 14, 11, 8, 5, 2],
            [18, 15, 12, 9, 6, 3]
        ])
        nrows, ncols = 3, 6

    # cage 3 layout - 4 cams
    elif n_cams == 4:
        
        # grab the cam_ids for the 4 cams
        # and insert them in this 2 x 2 pattern
        rows = np.array([
            [cam_ids[0], cam_ids[1]],
            [cam_ids[2], cam_ids[3]]
        ])
        nrows, ncols = 2, 2

        rows_idx = np.array([
            [4, 3],
            [2, 1]
        ])

    # load translation table
    fname_crop_table ="crop_table_cage_"+str(cage_id)+".yaml"
    crops = yaml.safe_load(open(fname_crop_table, 'r'))


    #print ("crop table: ", crops)
    #crop table:  {'cam1': [150, 1280, 0, 720], 'cam2': [150, 1280, 0, 720], 'cam3': [150, 1280, 0, 720], 'cam4': [100, 1180, 0, 720], 'cam5': [100, 1180, 0, 720], 'cam6': [100, 1180, 0, 720], 'cam7': [100, 1180, 0, 720], 'cam8': [100, 1180, 0, 720], 'cam9': [100, 1180, 0, 720], 'cam10': [100, 1180, 0, 720], 'cam11': [100, 1180, 0, 720], 'cam12': [100, 1180, 0, 720], 'cam13': [100, 1180, 0, 720], 'cam14': [100, 1180, 0, 720], 'cam15': [100, 1180, 0, 720], 'cam16': [0, 1180, 0, 720], 'cam17': [0, 1180, 0, 720], 'cam18': [0, 1180, 0, 720]}

    # ok need to figure out the fully frame size based on these crops and the rows that they will be appended above
    vid_width, vid_height = get_video_size(nrows, 
                                           ncols,
                                           rows_idx,
                                           crops)
    
    print ("final video size before shrink: ", vid_width, "x", vid_height)
    
    if (shrink_factor > 1):
        vid_width //= shrink_factor
        vid_height //= shrink_factor
        #print ("shrinking final video to: ", vid_width, "x", vid_height)

    #
    #################################################################
    #    
    frame_height = resolution[1]        # pixels
    frame_width = resolution[0]       # pixels
    channels = 3               # RGB
    if shrink_factor > 1:
        # need to divide by shrink factor for both height and width but round to nearest integer not floor
        frame_height = int(frame_height / shrink_factor+0.5)
        frame_width = int(frame_width / shrink_factor+0.5)
        #print ("shrinking video frames to: ", frame_height, "x", frame_width)

    # make a blank default image
    frame_blank = np.zeros((frame_height, frame_width, channels), dtype=np.uint8)
    print ("frame blank shape: ", frame_blank.shape)
    frame_size_bytes = frame_height * frame_width * channels  # 1280 * 768 * 3 = 2,359,296 bytes


    # we should make the final video size a multiple of the frame size -
    # we don't crop so much for 3D vids
    full_vid_height = frame_height * nrows
    full_vid_width = frame_width * ncols

    #

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
                          (full_vid_width, full_vid_height)
                          #(frame_width * ncols, frame_height * nrows)
                          )

    # 
    print ("final video size after shrink: ", full_vid_width, "x", full_vid_height)

    # ... hour:  16  ... minute:  17
    # Final video size: 4056 x 3040
    # final video size before shrink:  4056 x 3040
    # frame blank shape:  (152, 202, 3)
    # final video size after shrink:  404 x 304

    # seems based on input the width and ehigh mighb be swtiched? 
    # so we need to swap the width and height
    #full_vid_width, full_vid_height = full_vid_height, full_vid_width


    # we actually want to grab freeze frames from non-recorded cameras - so we don't reinitialize this
    for i in trange(0,frame_rate * duration, 
                    frame_subsample):
        #
        frame_all_cams_rows = [[],[],[]]

    
        # loop over cameras and grab a frame from each
        for ctr_cam, cam in enumerate(cam_ids):
            # find the filename for this camera and bin
            fname_frame = os.path.join(ram_disk_dir,
                                    str(cam) + "_shrink_" + str(shrink_factor) +
                                    "_hour_" + hour_start + "_minute_"+
                                    str(minute) + "_clean.bin")
            #
            #print ("fname_frame: ", fname_frame)
            
            #
            if os.path.exists(fname_frame):
                
                # we need to index into this file to the current frame i using framesize-bytes
                with open(fname_frame, 'rb') as f:
                    f.seek(i * frame_size_bytes)

                    frame_data = f.read(frame_size_bytes)
                    if len(frame_data) != frame_size_bytes:
                        # leave the previous frame in place
                        continue
                    
                    
                   # frame = np.frombuffer(frame_data, 
                   #                         dtype=np.uint8).reshape((frame_height, 
                   #                                                 frame_width, 
                   #                                                 channels))

                    #
                 
                    try:
                        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((frame_height, 
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
            # if ctr_cam == 1:
            #         print ("frame shape: ", frame.shape)


            #
            #print ("frame shape: ", frame.shape)
            # find row of the camera
            result = np.where(rows == cam)
            #print ("result", result)
            row_index = result[0][0] 

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

        if flip_vertical_flag:
            frame_all_cams_blank = np.flipud(frame_all_cams_blank)
       
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

