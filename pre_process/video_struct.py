import numpy as np
import random
import os
import pickle


class Video_ori(object):
    def __init__(self, lst_paths, label, video_name=None):
        self.lst_frame_paths = np.array(lst_paths)
        self.label = label
        self.num_frames = len(lst_paths)
        self.video_name = video_name
        self.cursor = 0


class Subject_ori(object):
    def __init__(self, label):
        self.label = label
        self.lst_video = []
        self.lst_video_no = []
        self.cursor = 0

    def add_video(self, video, no=0):
        self.lst_video.append(video)
        self.lst_video_no.append(no)


def save(lst, path):
    with open(path, 'wb') as fp:
        pickle.dump(lst, fp)


def get_dataset_struct(data_dir, save_dir, frames_thr=0, vides_thr=0):
    lst_subs = os.listdir(data_dir)
    cur_label = 0
    num_frames_per_video = []
    num_videos_per_subject = []
    lst_subjects = []
    for sub in lst_subs:
        print('subject is ' + sub)
        sub_struct = Subject_ori(label=cur_label)
        lst_videos = os.listdir(os.path.join(data_dir, sub))
        num_videos_per_subject.append(len(lst_videos))
        for video in lst_videos:
            video_dir = os.path.join(data_dir, sub, video)
            lst_paths = [os.path.join(sub, video, file) for file in os.listdir(video_dir) if file.endswith('.jpg')]
            num_frames_per_video.append(len(lst_paths))

            if len(lst_paths) < frames_thr:
                continue
            v_struct = Video_ori(lst_paths, label=cur_label, video_name=os.path.join(sub, video))
            sub_struct.add_video(v_struct)

        if len(sub_struct.lst_video)>= vides_thr:
            lst_subjects.append(sub_struct)
            cur_label += 1

    save(lst_subjects, os.path.join(save_dir, 'lst_subjects.txt'))
    save(num_frames_per_video, os.path.join(save_dir, 'num_frames_per_video.txt'))
    save(num_videos_per_subject, os.path.join(save_dir, 'num_videos_per_subject.txt'))
    return lst_subjects










