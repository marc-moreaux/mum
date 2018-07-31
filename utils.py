import face_recognition as frec
import matplotlib.pyplot as plt
import cv2
import moviepy
from  moviepy.editor import VideoFileClip
import numpy as np
import os
import glob
import scipy
import scipy.misc
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



def ends_with(str, ext):
    for e in ext:
        if str.endswith(e):
            return True
    return False


def get_ext_from_path(root, extentions=['.JPG', '.jpg']):
    img_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in [f for f in filenames if ends_with(f, extentions)]:
            img_paths.append(os.path.join(dirpath, filename))
    return img_paths


def reduce_image(img, base_width=500):
    img_width = float(img.shape[0])
    if img_width > base_width:
        wpercent = (base_width / img_width)
        hsize = int((float(img.shape[1]) * float(wpercent)))
        img = scipy.misc.imresize(img, (base_width, hsize))
    return img


def get_faces_encoding_from_dir(face_dir, debug=False):
    '''Get face encodings
    '''
    face_encodings = []
    print glob.glob(face_dir + '*.png')
    for f_path in glob.glob(face_dir + '/*.png'):
        try:
            img = frec.load_image_file(f_path)
            img = reduce_image(img)
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
    face_locations = frec.face_locations(img, 1, 'cnn')
    for loc in face_locations:
        t, r, b, l = loc
        face_img = img[t:b, l:r].copy()
        face_imgs.append(face_img)
    return face_imgs


def get_faces_from_path(img_path):
    '''Return an array of faces from an img path
    '''
    img = frec.load_image_file(img_path)
    img = reduce_image(img)
    faces = get_faces(img)
    return faces


def mum_in_image(img, mum_encodings, debug=False, append_encoding=True, precision=.4):
    img = reduce_image(img)
    faces_loc = frec.face_locations(img, 1, 'cnn')
    img_encodings = frec.face_encodings(img, faces_loc)
    for encoding, loc in zip(img_encodings, faces_loc):
        is_match = frec.compare_faces(mum_encodings, encoding, precision)
        if True in is_match:
            if append_encoding:
                mum_encodings.append(encoding)
            if debug is True:
                t, r, b, l = loc
                plt.imshow(img[t:b, l:r])
                plt.show()
            return True
    return False


def find_mum_in_video(video_path, found_video_paths, mum_encodings, fps=2):
    found = [video_path, ]
    try:
        clip = VideoFileClip(video_path)
        for i, frame in enumerate(clip.iter_frames(fps=fps)):
            try:
                if mum_in_image(frame, mum_encodings, append_encoding=False):
                    found.append(i/float(fps))
            except:
                print('Stg went wrong with a frame of {}'.format(video_path))
    except:
        print('Stg went wrong with the full file {}'.format(video_path))


    if len(found) > 1:
        found_video_paths.append(found)
        f = open("mum_video_paths.txt","a+")
        f.write(str(found) + '\n')
        f.close()
    
    print 'finished with' + video_path


def find_mum_in_image(image_path, found_image_paths, mum_encodings, precision=.4):
    try:
        img = frec.load_image_file(image_path)
        img = reduce_image(img)
        if mum_in_image(img, mum_encodings, precision=precision):
            if found_image_paths is not None:
                found_image_paths.append(image_path)        
                f = open("mum_paths.txt","a+")
                f.write(image_path + '\n')
                f.close()
    except:
        print('Stg went wrong with {}'.format(image_path))


def load_mum_paths(mum_paths='mum_paths'):
    f = open("mum_paths.txt","r")
    paths = f.readlines()
    f.close()
    paths = [p[:-1] for p in paths]
    return paths


def read_video_description(video_descriptions='mum_video_paths.txt', export_path='/tmp/videos/'):
    
    if not os.path.isdir(export_path):
        os.mkdir(export_path)
    
    with open(video_descriptions, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line[:-1]
        line = eval(line)
        video_path = line[0]
        times = np.array(line[1:])
        seqs = times_to_seq2(times, 6)
        print video_path, '\n', times, '\n', seqs, '\n'
        for i, (start, end) in enumerate(seqs):
            clip_name = video_path.split('/')[-1]
            clip_name = '.'.join(clip_name.split('.')[:-1])
            clip_name = clip_name + str(i) + '.mp4'
            clip = []
            clip.append(VideoFileClip(video_path))
            end = min(clip[0].duration, end)
            clip.append(clip[-1].resize(height=480))
            clip.append(clip[-1].subclip(start, end))
            clip[-1].write_videofile(os.path.join(export_path, clip_name))
            for _clip in clip:  # mandatory to flush memory :/
                _clip.close()


def times_to_seq(times, max_padding=5, time_thld=2, time_augment=3):
    '''
    max_padding: Accepted padding (in sec) between two recognized images
    time_thld: the minimum time between start and end
    time_augment: augment sequence by x sec'''
    cur_idx = 0
    start = times[0]
    end = times[0]

    seqs = []
    while cur_idx + 1 < len(times):
        next_idx = cur_idx + 1
        if times[next_idx] - times[cur_idx] < max_padding:
            end = times[cur_idx]
            if next_idx + 1 == len(times):
                start = max(0, start - time_augment)
                end = min(end + time_augment, times[-1])
                seqs.append((start, end))
        else:
            if end - start > time_thld:
                start = max(0, start - time_augment)
                end = min(end + time_augment, times[-1])
                seqs.append((start, end))
            start = times[next_idx]
        cur_idx += 1
    
    return seqs


def times_to_seq2(times, time_augment=5):
    '''
    time_augment: augment sequence by x sec
    '''
    cur_idx = 0
    start = times[0]
    end = times[0]
    seqs = []

    def append(start, end):
        ''' append a time segment to the <seqs> list
        '''
        start = max(0, start - time_augment)
        # end = min(end + time_augment, times[-1])
        end = end + time_augment
        seqs.append((start, end))

    while cur_idx + 1 < len(times):
        next_idx = cur_idx + 1

        # Check if time_indexes are close
        if times[next_idx] - times[cur_idx] < time_augment * 2:
            end = times[next_idx]
        # If not, append a time segment
        else:
            append(start, end)
            start = times[next_idx]
            end = times[next_idx]

        # End condition
        if next_idx + 1 == len(times):
            append(start, end)

        cur_idx += 1
    
    return seqs
