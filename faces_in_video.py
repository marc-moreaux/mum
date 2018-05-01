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

mum_dir = '/home/mmoreaux/work/mum/faces_to_reco/'
mum_encodings = utils.get_faces_encoding_from_dir(mum_dir, debug=False)

print(len(mum_encodings))

#%%
import glob
reload(utils)

picture_paths = glob.glob('/media/mmoreaux/Data/photos/photos2/2001/**/*.jpg')
video_paths = []

found_image_paths = []
found_video_paths = []


for image_path in picture_paths[:30]:
    utils.find_mum_in_image(image_path, found_image_paths, mum_encodings)

#%%
print found_image_paths
print len(found_image_paths)
for img in found_image_paths[:30]:
    img = face_recognition.load_image_file(img)
    plt.imshow(img)
    plt.show()