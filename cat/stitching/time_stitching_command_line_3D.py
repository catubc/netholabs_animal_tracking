import numpy as np
import os
from utils import make_video, process_cams, get_ramdisk_dir, get_ramdisk_dir_non_memory
import yaml

#############################################
#############################################
#############################################
input_dir = "/home/cat/Downloads/3D_Vids_to_be_stiched/9_12/"
output_dir = "/home/cat/Downloads/3D_Vids_to_be_stiched/9_12/"
input_dir = '/home/cat/Downloads/Cage 1 Staging/Cage 1 Cameras/'
output_dir = '/home/cat/Downloads/Cage 1 Staging/Cage 1 Cameras/'

input_dir = "/home/cat/Downloads/newdata2/"
output_dir = "/home/cat/Downloads/newdata2/"
# input_dir = "/home/cat/Downloads/3D_Vids_to_be_stiched/25_28/"
# output_dir = "/home/cat/Downloads/3D_Vids_to_be_stiched/25_28/"


#
fname_config = os.path.join(output_dir,'config.yaml')
config = yaml.safe_load(open(fname_config, 'r'))

dates = config['dates']
print ("Dates to process: ", dates)

##### PARAMS OF DATA SETS #######
hour_start = 0                                     # start at midnight
minutes_default = np.arange(hour_start*60, 24*60)  # which minutes of the day to process
first_day_hour_start = 14
first_day_minute_start = 25

#
shrink_factor = 1
frame_subsample = 1
cage_id = 3
n_cams = 4
cam_ids = config['cam_ids']

##### PARAMS OF DATA GENERATION
skip_regeneration = False  # if True, delete old bin files before making new ones
parallel_flag = False
n_cores = 8
delete_bins_flag = False
build_video_only = False
use_ramdisk = False
flip_vertical_flag = True


# first make ramdisk dir
if use_ramdisk:
    ram_disk_dir = get_ramdisk_dir()
else:
    ram_disk_dir = get_ramdisk_dir_non_memory()

# loop over all dates provided
for date in dates:

    #
    if False:
        if date == dates[0]:
            minutes = np.arange(first_day_hour_start*60, 24*60)
        else:
            minutes = minutes_default
    else:
        minutes = [first_day_hour_start*60+first_day_minute_start,
                    first_day_hour_start*60+first_day_minute_start+2]

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
                                      "date_"+ str(date) + "_"
                                      "hour_" + str(hour_start) + "_" +
                                      "minute_" +str(minute)+'.avi')

        #
        if os.path.exists(fname_combined) and skip_regeneration:
            print ("Video exists for Minute ", str(minute), "...skipping...")
            continue

        if True:
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
            overwrite_existing = True
            make_video(config,
                        flip_vertical_flag,
                       input_dir,
                        ram_disk_dir,
                        date,
                        minute,
                        hour_start,
                        n_cams,
                        cam_ids,
                        fname_combined,
                        shrink_factor=shrink_factor,
                        frame_subsample=frame_subsample,
                        overwrite_existing=overwrite_existing,
                        delete_bins_flag = delete_bins_flag)
            
            #print ("***************************")
            print ('')

        else:
        #except:
            # dlete the whole ramdrive and continue
            os.system("rm -rf " + ram_disk_dir + "/*")

        #break

    
