#%%
import face_recognition
import matplotlib.pyplot as plt
import cv2
import moviepy
from  moviepy.editor import VideoFileClip
import numpy as np
import os
import utils
reload(utils)
import os
import os.path


mum_dir = './faces_to_reco/'
mum_encodings = utils.get_faces_encoding_from_dir(mum_dir, debug=False)
mum_paths = utils.load_mum_paths()
for image_path in mum_paths:
    utils.find_mum_in_image(image_path, None, mum_encodings)

print 'there is {} encodings of mum'.format(len(mum_encodings))

#%% images
reload(utils)

picture_paths = utils.get_ext_from_path('/media/moreaux/3tb_disk/Data_SSD/DisqueDur/photos/')
print ('searching in {} pictures'.format(len(picture_paths)))

found_image_paths = []
for image_path in picture_paths:
    utils.find_mum_in_image(image_path, found_image_paths, mum_encodings)


#%% videos
reload(utils)
video_paths = []
for root in [
    '/media/moreaux/3tb_disk/cam_maman',
    '/media/moreaux/3tb_disk/cam_maman_old',
    '/media/moreaux/3tb_disk/mes photos',
    '/media/moreaux/3tb_disk/Data_SSD',
    '/media/moreaux/Disk2/Dropbox']:
    video_paths += utils.get_ext_from_path(root, ['.mp4', 'mpg', '.MTS', '.mts', '.MP4'])
print ('searching in {} videos'.format(len(video_paths)))

found_video_paths = []
for video_path in video_paths:
    utils.find_mum_in_video(video_path, found_video_paths, mum_encodings)


#%% TEST
reload(utils)
utils.read_video_description()

