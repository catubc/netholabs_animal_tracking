import numpy as np
import os
from utils import make_video, process_cams, get_ramdisk_dir

#############################################
#############################################
#############################################
n_cams = 18
root_dir = "/home/cat/Downloads/data_stitching/cams/"
date = "2025_07_31"                               # date of the video
hour_start = "07"                                     # start at midnight
n_mins = 1440
shrink_factor = 10   # this shrinks the video along x and y axis by this factor; for now we use subsampling
frame_subsample = 4
cage_id = 1
cam_ids = np.arange(1, n_cams + 1)

#
skip_regeneration = False  # if True, delete old bin files before making new ones
parallel_flag = True
n_cores = 4
delete_bins_flag = True
build_video_only = False

# first make ramdisk dir
ram_disk_dir = get_ramdisk_dir()


# outer loop gooin over every minue of time 
for minute in range(n_mins):

    # this is to run full days
    if False:
        hour_start = str(minute // 60).zfill(2)
        minute = minute % 60

    # this is for debugging
    else:
        pass

    # save the combined frame as a video file
    fname_combined = os.path.join(root_dir,
                                  "hour_" + hour_start+
                                  "_minute_" +str(minute)+'.avi')

    #
    if os.path.exists(fname_combined):
        print ("Video exists for Minute ", str(minute), "...skipping...")
        continue

    #
    if build_video_only==False:
        process_cams(cam_ids,
                     n_cores,
                    root_dir,
                    ram_disk_dir,
                    cage_id,
                    date,
                    hour_start,
                    minute,
                    parallel_flag=parallel_flag,
                    shrink_factor=shrink_factor,
                    skip_regeneration=False)

    #################################################
    ############### Make video ######################
    #################################################
    # process #2 - here we make the mosaic 1 minute video based on the available files
    # we loop over all possible files 
    overwrite_existing = False
    make_video(root_dir,
               ram_disk_dir,
               date,
               minute,
               hour_start,
               n_cams,
               fname_combined,
               shrink_factor=shrink_factor,
               frame_subsample=frame_subsample,
               overwrite_existing=overwrite_existing,
               delete_bins_flag = delete_bins_flag)
    
    #print ("***************************")
    print ('')

    #break

    