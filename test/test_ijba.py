import os
import numpy as np
from IJB_A import read_IJB_A_img_path
import pickle
import eval
import h5py
import sklearn.metrics.pairwise as skp


def save_split(lst, path):
    with open(path, 'wb') as fp:
        pickle.dump(lst, fp)


def norm_l2(feat, eps=1e-10):
    feat = feat/np.sqrt(np.sum(np.multiply(feat, feat)) + eps)
    return feat


def intra_media(feat, mlabel):
    # media-pooling, refer to Eq. (8)
    if feat.shape[0] == 1:
        return feat, False
    idx_0 = np.where(mlabel == 0)[0].tolist()
    idx_1 = np.where(mlabel == 1)[0].tolist()

    if len(idx_1) == 0:
        return np.mean(feat, axis=0, keepdims=True), False
    if len(idx_0) == 0:
        return np.mean(feat, axis=0, keepdims=True), True

    feat_0 = feat[idx_0, :]
    feat_1 = feat[idx_1, :]
    mfeat_0 = np.mean(feat_0, axis=0, keepdims=True)
    mfeat_1 = np.mean(feat_1, axis=0, keepdims=True)
    return (mfeat_0 + mfeat_1)/2.0, True
    # return mfeat_1, True


def sofmax_match(feat_a, feat_b):
    # softmax score pooling
    sim_mat = skp.cosine_similarity(feat_a, feat_b)
    csim = 0
    hin = 21
    for beta in range(0, hin):
        sim_beta = np.exp(beta*sim_mat)
        sum = np.sum(sim_beta)
        sim_beta = sim_beta/sum
        w_sim = sim_mat * sim_beta
        csim += np.sum(w_sim)
    return csim/hin


def main():
    # settings
    net_type = 'resnet_34_coso'
    src_dir = '../data/IJB_A/deep_feat/'
    deep_feat_file = 'IJB_A_{}.h5'.format(net_type)
    split_data_dir = '../data/IJB_A/split_data'
    softmax_fusion = False


    np_tars = []
    fars = []
    for idx_split in range(1, 11):
        print('split %d' % idx_split)
        fars, tars = run_one_split(idx_split, split_data_dir=split_data_dir,
                                   deep_feat_file=deep_feat_file, src_dir=src_dir, softmax_fusion=softmax_fusion)
        if idx_split == 1:
            np_tars = np.expand_dims(np.array(tars), axis=0)
        else:
            np_tars = np.vstack([np_tars, np.expand_dims(np.array(tars), axis=0)])

    print('Average Result is')
    print('FAR:')
    print(fars)
    print('TAR:')
    print(np.mean(np_tars, axis=0))
    print('STD:')
    print(np.std(np_tars, axis=0))
    print('finished')


def run_one_split(idx_split, split_data_dir, deep_feat_file, src_dir, softmax_fusion):
    # idx_split = FLAGS.idx_split
    test_file = os.path.join(split_data_dir, 'lst_test_pairs_' + str(idx_split) + '.txt')
    lst_test_pairs = pickle.load(open(test_file, 'rb'))

    print('testing...')
    c_sim = np.zeros(shape=[len(lst_test_pairs)], dtype=np.float32)
    actual_issame = np.zeros(shape=[len(lst_test_pairs)], dtype=np.bool)
    cur_pair = 0

    hf = h5py.File(os.path.join(src_dir, deep_feat_file), 'r')

    for pair in lst_test_pairs:
        s1 = pair[3]
        vfea_a = hf[s1][:]

        s2 = pair[4]
        vfea_b = hf[s2][:]

        if softmax_fusion:
            c_sim[cur_pair] = sofmax_match(vfea_a, vfea_b)
        else:

            mfeat_a = np.mean(vfea_a, axis=0, keepdims=True)  # [1, dim]
            mfeat_b = np.mean(vfea_b, axis=0, keepdims=True)
            nfeat_a = norm_l2(mfeat_a)
            nfeat_b = norm_l2(mfeat_b)
            cos_d = np.dot(nfeat_b, np.transpose(nfeat_a))
            c_sim[cur_pair] = cos_d
        actual_issame[cur_pair] = pair[2]
        cur_pair += 1

    idx = np.where(c_sim > -2)[0].tolist()
    c_sim = c_sim[idx]
    actual_issame = actual_issame[idx]
    save_path = None
    fars, tars, thrs, FA, TA, auc, acc = eval.cal_far(c_sim, actual_issame, save_path=save_path)
    cur = 0
    for far in fars:
        print('tar=%f @ far=%f' % (tars[cur], far))
        cur += 1
    print('AUC = %f' % auc)
    print('Accuracy = %f' % acc)
    return fars, tars


if __name__ == '__main__':
   main()
