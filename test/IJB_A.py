import os
from pre_process.video_struct import Video_ori
import numpy as np
import pickle
from pre_process.video_struct import Subject_ori
import h5py as hp
import scipy.io as sio


def gen_list_path(data_dir):
    '''
    get ijb-a all lst_videos(templates)
    :param data_dir:
    :return:
    '''
    lst_subject_path = os.listdir(data_dir)#[file for file in os.listdir(data_dir) if file.endswith('.jpg')]
    dict_sub = {}
    lst_video_data = []
    lst_labels = []
    cur_id = 0
    for sub in lst_subject_path:
        has_key = sub in dict_sub.keys()
        if not has_key:
            label = cur_id
            dict_sub[sub] = label
            cur_id += 1
        else:
            label = dict_sub[sub]

        lst_tem = os.listdir(os.path.join(data_dir, sub))
        for tem in lst_tem:
            #lst_paths = [os.path.join(video, file) for file in os.listdir(sub_dir) if file.endswith('.jpg')]
            tem_dir = os.path.join(data_dir, sub, tem)
            lst_paths = [os.path.join(sub, tem, file) for file in os.listdir(tem_dir) if file.endswith('.jpg')]
            lst_video_data.append(Video_ori(lst_paths=lst_paths, label=label, video_name=tem))
            lst_labels.append(label)
    return lst_video_data, lst_labels


def save(lst, path):
    with open(path, 'wb') as fp:
        pickle.dump(lst, fp)


def read_IJB_A_img_path(split_dir, data_dir, idx_split):
    # from pre_process.video_struct import Subject
    # from pre_process.video_struct import Video
    train_file = os.path.join(split_dir, 'split' + str(idx_split), 'train_' + str(idx_split) + '.csv')
    cur_line = 0
    dict_templates = {}  # template: subject
    lst_subject_name = []
    for line in open(train_file):
        cur_line += 1
        if cur_line == 1:  # skip the first line
            continue
        lst_tmp = line.split(',')
        template_id = lst_tmp[0]
        subject_id = lst_tmp[1]
        lst_subject_name.append(subject_id)
        dict_templates.update({template_id: subject_id})

    unique_subjects = sorted(set(lst_subject_name), key=lst_subject_name.index)
    cur_label = 0
    lst_subject = list(dict_templates.values())
    lst_templates = list(dict_templates.keys())
    # read train split file
    lst_train_subject_multi = []
    lst_train_subject_single = []
    for sub in unique_subjects:
        one_sub = Subject_ori(label=cur_label)
        lst_idx = [i for i, x in enumerate(lst_subject) if x == sub]#the template
        for idx in lst_idx:
            tem = lst_templates[idx]
            tem_dir = os.path.join(data_dir, sub, tem)
            lst_paths = [os.path.join(sub, tem, file) for file in os.listdir(tem_dir) if file.endswith('.jpg')]
            one_video = Video_ori(lst_paths=lst_paths, label=cur_label, video_name=tem)
            one_sub.add_video(video=one_video)
        if len(lst_idx) == 1:
            lst_train_subject_single.append(one_sub)
        else:
            lst_train_subject_multi.append(one_sub)
        cur_label += 1

    # read meta data file
    meta_file = os.path.join(split_dir, 'split' + str(idx_split), 'verify_metadata_' + str(idx_split) + '.csv')
    cur_line = 0
    dict_all_templates = {}
    for line in open(meta_file):
        cur_line += 1
        if cur_line == 1:
            continue
        lst_tmp = line.split(',')
        template_id = lst_tmp[0]
        subject_id = lst_tmp[1]
        dict_all_templates.update({template_id: subject_id})
    # read test split file
    test_file = os.path.join(split_dir, 'split' + str(idx_split), 'verify_comparisons_' + str(idx_split) + '.csv')
    lst_test_pairs = []
    set_test_tem = set()
    for line in open(test_file):
        lst_tmp = line.split(',')
        tem_a = lst_tmp[0].replace('\n', '')
        tem_b = lst_tmp[1].replace('\n', '')

        sub_a = dict_all_templates[tem_a]
        sub_b = dict_all_templates[tem_b]

        set_test_tem.add(os.path.join(sub_a, tem_a))
        set_test_tem.add(os.path.join(sub_b, tem_b))

        tem_dir = os.path.join(data_dir, sub_a, tem_a)
        lst_paths = [os.path.join(sub_a, tem_a, file) for file in os.listdir(tem_dir) if file.endswith('.jpg')]
        video_a = Video_ori(lst_paths=lst_paths, label=-1, video_name=tem_a)

        tem_dir = os.path.join(data_dir, sub_b, tem_b)
        lst_paths = [os.path.join(sub_b, tem_b, file) for file in os.listdir(tem_dir) if file.endswith('.jpg')]
        video_b = Video_ori(lst_paths=lst_paths, label=-1, video_name=tem_a)

        if sub_a == sub_b:
            pair_label = True
        else:
            pair_label = False
        lst_test_pairs.append([video_a, video_b, pair_label, tem_a, tem_b])

    print('Have read IJB_A dataset')
    return lst_train_subject_multi, lst_train_subject_single, lst_test_pairs, set_test_tem

def read_IJB_A(split_dir, data_dir, idx_split):
    from pre_process.video_struct import Subject_feat
    from pre_process.video_struct import Set_feat

    # read train split file
    lst_train_subject_multi = []
    lst_train_subject_single = []
    lst_subject_name = []
    train_file = os.path.join(split_dir, 'split' + str(idx_split), 'train_' + str(idx_split)+'.csv')
    cur_line = 0
    dict_templates = {}#template: subject
    for line in open(train_file):
        cur_line += 1
        if cur_line == 1:#skip the first line
            continue
        lst_tmp = line.split(',')
        template_id = lst_tmp[0]
        subject_id = lst_tmp[1]
        lst_subject_name.append(subject_id)
        dict_templates.update({template_id: subject_id})
        #if template_id not in lst_template_na

    unique_subjects = sorted(set(lst_subject_name), key=lst_subject_name.index)
    cur_label = 0
    lst_subject = list(dict_templates.values())
    lst_templates = list(dict_templates.keys())

    for sub in unique_subjects:
        one_sub = Subject_feat(label=cur_label, orgin_label=sub)
        lst_idx = [i for i, x in enumerate(lst_subject) if x==sub]
        for idx in lst_idx:
            #np.load(os.path.join(FLAGS.logs_base_dir, 'YTF_dp_feat_' + str(FLAGS.idx_fold) + '.npy'))
            feat = np.load(os.path.join(data_dir, str(lst_templates[idx]) + '.npy'))
            one_set = Set_feat(feat=feat, label=cur_label, set_name=lst_templates[idx])
            one_sub.add_set(one_set)
        if len(lst_idx)==1:
            lst_train_subject_single.append(one_sub)
        else:
            lst_train_subject_multi.append(one_sub)
        cur_label += 1


    # read meta data file
    meta_file = os.path.join(split_dir, 'split' + str(idx_split), 'verify_metadata_' + str(idx_split)+'.csv')
    cur_line = 0
    dict_all_templates = {}
    for line in open(meta_file):
        cur_line += 1
        if cur_line == 1:
            continue
        lst_tmp = line.split(',')
        template_id = lst_tmp[0]
        subject_id = lst_tmp[1]
        dict_all_templates.update({template_id: subject_id})
    # read test split file
    test_file = os.path.join(split_dir, 'split' + str(idx_split), 'verify_comparisons_' + str(idx_split)+'.csv')
    lst_test_pairs = []
    for line in open(test_file):
        lst_tmp = line.split(',')
        id_a = lst_tmp[0].replace('\n', '')
        id_b = lst_tmp[1].replace('\n', '')
        feat_a = np.load(os.path.join(data_dir, str(id_a) + '.npy'))
        feat_b = np.load(os.path.join(data_dir, str(id_b) + '.npy'))
        if dict_all_templates[id_a] == dict_all_templates[id_b]:
            pair_label = True
        else:
            pair_label = False
        lst_test_pairs.append([feat_a, feat_b, pair_label, id_a, id_b])

    print('Have read IJB_A dataset')
    return lst_train_subject_multi, lst_train_subject_single, lst_test_pairs


def run_rename_file(data_dir='D:\\yirong\\data\\IJB-A\\resnet50_feat\\deep_feat'):
    lst_tem = os.listdir(data_dir)
    for tem in lst_tem:
        strs = tem.split('_')
        os.rename(os.path.join(data_dir, tem), os.path.join(data_dir, strs[1]))


def read_mat_to_hf5(data_dir='D:\\yirong\\data\\IJB-A\\resnet50_ft_feat_ant\\'):
    lst_tem = os.listdir(os.path.join(data_dir, 'deep_feat'))
    hf_save_path = 'resnet50_ft_feat_ant.h5'
    hf = hp.File(os.path.join(data_dir, hf_save_path), 'w')
    for tem in lst_tem:
        fid = hp.File(os.path.join(data_dir, 'deep_feat', tem))
        vfea = np.array(fid['vfea'])
        hf.create_dataset(tem, data=vfea)
    hf.close()




def main():
    read_mat_to_hf5()
    # run_rename_file()
    # read IJB_A dataset to lst_video_data structure
    # save_dir = '../data/IJB_A/'
    # data_dir = 'D://yirong//data//IJB-A//IJB-A_img_auto//img_100x100'
    # lst_video_data, lst_label = gen_list_path(data_dir=data_dir)
    # file_name = 'lst_template' + '.txt'
    # save(lst_video_data, os.path.join(save_dir, file_name))
    # file_name = 'lst_template_label' + '.txt'
    # save(lst_label, os.path.join(save_dir, file_name))


    print('finished')

if __name__ == '__main__':
    main()