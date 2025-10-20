import numpy as np
import os
from utils import make_video, process_cams, get_ramdisk_dir

#############################################
#############################################
#############################################
input_dir = "/home/cat/Downloads/data_stitching/cams/"
input_dir = '/mnt/data2/netholabs/'
output_dir = "/mnt/ssd/"


###### INSERT DATES OF VIDEOS #######
###### INSERT DATES OF VIDEOS #######
dates = [                              # date of the video
    "2025_09_17",
    "2025_09_18",
    "2025_09_19",
    "2025_09_20",
    "2025_09_21"
]


##### PARAMS OF DATA SETS #######
hour_start = 0                                     # start at midnight
minutes_default = np.arange(hour_start*60, 24*60)  # which minutes of the day to process
first_day_hour_start = 20

#
shrink_factor = 10
frame_subsample = 4
cage_id = 1
n_cams = 18
cam_ids = np.arange(1, n_cams + 1)

##### PARAMS OF DATA GENERATION
skip_regeneration = False  # if True, delete old bin files before making new ones
parallel_flag = True
n_cores = 8
delete_bins_flag = True
build_video_only = False

# first make ramdisk dir
ram_disk_dir = get_ramdisk_dir()

# loop over all dates provided
for date in dates:

    #
    if date == dates[0]:
        minutes = np.arange(first_day_hour_start*60, 24*60)
    else:
        minutes = minutes_default

    # outer loop gooin over every minue of time 
    for minute in minutes:

        #
        hour_start = str(minute // 60).zfill(2)
        minute = minute % 60
        
        #
        print ("... hour: ", hour_start, " ... minute: ", minute)

        # save the combined frame as a video file
        fname_combined = os.path.join(output_dir,
                                        "cage_" + str(cage_id) + "_" +
                                      "date_"+ date + "_"
                                      "hour_" + hour_start + "_" +
                                      "minute_" +str(minute)+'.avi')

        #
        if os.path.exists(fname_combined):
            print ("Video exists for Minute ", str(minute), "...skipping...")
            continue

        try:
            #
            if build_video_only==False:
                process_cams(cam_ids,
                            n_cores,
                            input_dir,
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
            make_video(input_dir,
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

        except:
            # dlete the whole ramdrive and continue
            os.system("rm -rf " + ram_disk_dir + "/*")

        #break

    
