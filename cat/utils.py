import numpy as np
import os
import glob
import cv2
import shutil  # <-- needed for copying


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

#
def decompress_video(minute, 
                     fname,
                     root_dir,
                     cam,
                     time_stamps_binned,
                     unix_time_to_start_of_minute,
                     shrink_factror=1,
                     overwrite_existing=False):
    
    ''' use OpenCV or something fast to load the video file or uncompress it.
        Not clear yet if we should just save to disk, but probably,
    '''
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
    fname_video_current_minute_clean= os.path.split(fname)[0]+"/minute_" + str(minute) + "_clean.bin"
    fname_video_current_minute_temp = os.path.split(fname)[0]+"/minute_" + str(minute) + "_temp.bin"
    fname_video_next_minute_temp = os.path.split(fname)[0]+"/minute_" + str(minute+1) + "_temp.bin"
    
    # shrink based on the shrink factor
    if shrink_factror > 1:
        frame_height //= shrink_factror
        frame_width  //= shrink_factror
        print ("shrinking video frames to: ", frame_height, "x", frame_width)
    
    # make blank frame 
    frame_size_bytes = frame_height * frame_width * channels
    blank_frame = np.zeros((frame_height, frame_width, channels), dtype=np.uint8)

    # we offset the loaded time steps to the start of the current minute
    # so we can then use a 6000 bin fixed size video structure 
    times_relative_to_min_start = time_stamps_binned - unix_time_to_start_of_minute
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
        print ("... found temp video file for current minute: ", 
               fname_video_current_minute_clean)
        file_size = os.path.getsize(fname_video_current_minute_temp)
        frames_already_written = file_size // frame_size_bytes
        print(f"File already exists: {frames_already_written} frames found.")

        # let's make a copy of the temp file to the current minute
        # we can now append to the fname_current_minute as required - and 
        #shutil.copy(fname_video_current_minute_clean, fname_video_current_minute_temp)

    else:
        frames_already_written = 0
        print("File does not exist yet.")

    # need to check that if there is a full video in place already then we can exit safely
    if frames_already_written >= 6000:
        print ("... this minute of data already fully decompressed, exiting ...")
        print (" --- THIS CASE SHOULDNT HAPPNE... to check...")
        return

    # we need to make sure that the first time stamp is 0
    # so we now loop over the video and the raw frames 
    # we fill blanks until we reach the next time stamp

    # use open cv to laod video
    import cv2
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
            if shrink_factror > 1:
                # let's use simple subsampling
                if True:
                    frame = frame[::shrink_factror, ::shrink_factror, :]
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
    print ("... clean exit, renaming current minute temp file to clean file ...")
    shutil.move(fname_video_current_minute_temp, fname_video_current_minute_clean)
    
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
            if shrink_factror > 1:
                # let's use simple subsampling
                if True:
                    frame = frame[::shrink_factror, ::shrink_factror, :]
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

    print ("# UNIQUE frames written to next min video", n_unique_frames_written,
              ", n_frames_read: ", n_frames_read)

#
def make_video(root_dir,
                minute,
                n_cams,
                fname_combined,
                shrink_factor=1,
                overwrite_existing = False
                ):
    
    from tqdm import trange
    ''' make a video from the available frames for this minute '''
    
        
    frame_height = 720        # pixels
    frame_width = 1280       # pixels
    channels = 3               # RGB
    if shrink_factor > 1:
        frame_height //= shrink_factor
        frame_width  //= shrink_factor
        print ("shrinking video frames to: ", frame_height, "x", frame_width)

    #
    frame_size_bytes = frame_height * frame_width * channels  # 1024 * 768 * 3 = 2,359,296 bytes

    if os.path.exists(fname_combined) and overwrite_existing==False:
        print ("... combined video file already exists: ", fname_combined)
        return

    rows, cols = 3, 6
    frame_all_cams_blank = np.zeros((frame_height * rows, frame_width * cols, channels), dtype=np.uint8)

    print ("creating combined video file: ", fname_combined)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(fname_combined, fourcc, 30.0, (frame_width * cols, frame_height * rows))

    # we actually want to grab freeze frames from non-recorded cameras - so we don't reinitialize this
    for i in trange(6000):
    
        # loop over cameras and grab a frame from each
        for cam in range(1,n_cams+1,1):
            # find the filename for this camera and bin
            fname_frame = os.path.join(root_dir,
                                    str(cam),
                                    "minute_"+
                                    str(minute) + "_clean.bin")
            
            #
            if os.path.exists(fname_frame):
                # 
                print ("making video with camera")

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
                        print ("i: ", i)
                        print ("cam: ", cam)
                        print ("fname: ", fname_frame)
                        print ("frame_size_bytes: ", frame_size_bytes)

                # 
                # adding frame from camera:  2  to row:  2  col:  0  at position:  144 216 0 128
                # adding frame from camera:  3  to row:  0  col:  0  at position:  0 72 0 128
                                
                col = 5 - ((cam - 1) // 3)  # 6 total columns → index 0..5 reversed
                row = (cam - 1) % 3  # 0=top, 1=middle, 2=bottom in OpenCV coords

                r0 = row * frame_height
                r1 = r0 + frame_height
                c0 = col * frame_width
                c1 = c0 + frame_width

                frame_all_cams_blank[r0:r1, c0:c1, :] = frame


                #
        #         print ("adding frame from camera: ", cam, 
        #                " to row: ", row, " col: ", col, 
        #                " at position: ", r0, r1, c0, c1)

        # return
            
        # now write the combined frame to the video file
        out.write(frame_all_cams_blank)
    # this is probably coming at the end. not sure exactly
    out.release()
    print ('finished writing combined video file: ', fname_combined)