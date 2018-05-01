import face_recognition as frec
import matplotlib.pyplot as plt
import cv2
import moviepy
from  moviepy.editor import VideoFileClip
import numpy as np
import os
import glob


def get_faces_encoding_from_dir(face_dir, debug=False):
    '''Get face encodings
    '''
    face_encodings = []
    print glob.glob(face_dir + '*.png')
    for f_path in glob.glob(face_dir + '/*.png'):
        try:
            img = frec.load_image_file(f_path)
            faces_loc = frec.face_locations(img)
            img_encodings = frec.face_encodings(img, faces_loc)
            face_encodings.append(frec.face_encodings(img)[0])

            if debug is True:
                loc = faces_loc[0]
                t, r, b, l = loc
                plt.imshow(img[t:b, l:r])
                plt.show()
            
        except IndexError:
            print "I wasn't able to locate any faces in ", _file
        
    return face_encodings


def get_faces(img):
    '''return list of faces seen on img
    '''
    face_imgs = []
    face_locations = frec.face_locations(img, 1, 'hog')
    for loc in face_locations:
        t, r, b, l = loc
        face_img = img[t:b, l:r].copy()
        face_imgs.append(face_img)
    return face_imgs


def get_faces_from_path(img_path):
    '''Return an array of faces from an img path
    '''
    img = frec.load_image_file(img_path)
    faces = get_faces(img)
    return faces


def mum_in_image(img, mum_encodings):
    faces_loc = frec.face_locations(img)
    img_encodings = frec.face_encodings(img, faces_loc)
    for encoding, loc in zip(img_encodings, faces_loc):
        is_match = frec.compare_faces(mum_encodings, encoding, .9)
        if True in is_match:
            #mum_encodings.append(encoding)
            t, r, b, l = loc
            plt.imshow(img[t:b, l:r])
            plt.show()
            return True
    return False


def find_mum_in_video(video_path, found_video_paths, mum_encodings, fps=2):
    clip = VideoFileClip(video_path)
    found = [video_path, ]
    for i, frame in enumerate(clip.iter_frames(fps=fps)):
        if mum_in_image(frame, mum_encodings):
            found.append(i)
    if len(found) > 1:
        found_video_paths.append(found)


def find_mum_in_image(image_path, found_image_paths, mum_encodings):
    img = frec.load_image_file(image_path)
    if mum_in_image(img, mum_encodings):
        found_image_paths.append(image_path)


