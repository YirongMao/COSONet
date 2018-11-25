import numpy as np
from scipy import interpolate
import sklearn.metrics.pairwise as skp


def norm_l2(feat, eps=1e-10):
    feat = feat/np.sqrt(np.sum(np.multiply(feat, feat)) + eps)
    return feat


def cal_pairs_acc(embeddings, actual_issame):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    fold_size = embeddings1.shape[0]
    sim = np.zeros(shape=[fold_size], dtype=np.float32)
    for i in range(embeddings1.shape[0]):
        mean_a = embeddings1[i, :]
        mean_b = embeddings2[i, :]
        # tmp = np.matmul(mean_a.T, mean_b) / (np.linalg.norm(mean_a, ord=2) * np.linalg.norm(mean_b, ord=2))
        mean_a = np.reshape(mean_a, (1, -1))
        mean_b = np.reshape(mean_b, (1, -1))
        # tmp = skp.cosine_similarity(X=mean_a, Y=mean_b, dense_output=True)
        # tmp = tmp[0][0]

        nfeat_a = norm_l2(mean_a)
        nfeat_b = norm_l2(mean_b)
        cos_d = np.dot(nfeat_b, np.transpose(nfeat_a))
        sim[i] = cos_d

    min_sim = sim.min()
    max_sim = sim.max()
    thr = np.linspace(min_sim, max_sim, num=1000)
    FA = np.zeros(thr.shape, dtype=np.float32)
    TN = np.zeros(thr.shape, dtype=np.float32)
    TA = np.zeros(thr.shape, dtype=np.float32)
    acc = np.zeros(thr.shape, dtype=np.float32)
    idx_pos = np.where(actual_issame == True)[0].tolist()
    idx_neg = np.where(actual_issame == False)[0].tolist()
    # idx_pos = np.array(range(250)).tolist()
    # idx_neg = np.array(range(250, 500)).tolist()
    # import pdb
    # pdb.set_trace()
    num_pos = len(idx_pos)
    num_neg = len(idx_neg)
    cur = 0
    for cur_thr in thr:
        # pos_set = sim[idx_pos]
        num_ta = np.where(sim[idx_pos] > cur_thr)[0].shape[0]
        num_fa = np.where(sim[idx_neg] > cur_thr)[0].shape[0]
        num_tn = np.where(sim[idx_neg] < cur_thr)[0].shape[0]

        FA[cur] = num_fa / num_neg
        TA[cur] = num_ta / num_pos
        acc[cur] = (num_ta + num_tn) / (num_neg + num_pos)
        cur += 1
    order_fa = np.argsort(FA)
    FA = FA[order_fa]
    TA = TA[order_fa]
    auc = 0
    for i in range(FA.shape[0]):
        if i == 0:
            step = FA[i] - 0.00
        else:
            step = FA[i] - FA[i - 1]
        auc += step * TA[i]

    fars = [0.001, 0.005, 0.01, 0.05, 0.1]
    tars = []
    for far in fars:
        idx_target = np.argmin(np.abs(FA - far))
        tars.append(TA[idx_target])
    print(str(tars) + str(fars))
    print('AUC = %f' % auc)
    print('Accuracy = %f' % np.max(acc))


def YTF_eval(embeddings, actual_issame):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    fold_size = embeddings1.shape[0]
    sim = np.zeros(shape=[fold_size], dtype=np.float32)
    for i in range(embeddings1.shape[0]):
        mean_a = embeddings1[i, :]
        mean_b = embeddings2[i, :]
        # tmp = np.matmul(mean_a.T, mean_b) / (np.linalg.norm(mean_a, ord=2) * np.linalg.norm(mean_b, ord=2))
        mean_a = np.reshape(mean_a, (1, -1))
        mean_b = np.reshape(mean_b, (1, -1))
        tmp = skp.cosine_similarity(X=mean_a, Y=mean_b, dense_output=True)
        tmp = tmp[0][0]
        sim[i] = tmp

    min_sim = sim.min()
    max_sim = sim.max()
    thr = np.linspace(min_sim, max_sim, num=1000)
    FA = np.zeros(thr.shape, dtype=np.float32)
    TN = np.zeros(thr.shape, dtype=np.float32)
    TA = np.zeros(thr.shape, dtype=np.float32)
    acc = np.zeros(thr.shape, dtype=np.float32)
    idx_pos = np.where(actual_issame == True)[0].tolist()
    idx_neg = np.where(actual_issame == False)[0].tolist()
    # idx_pos = np.array(range(250)).tolist()
    # idx_neg = np.array(range(250, 500)).tolist()
    # import pdb
    # pdb.set_trace()
    num_pos = len(idx_pos)
    num_neg = len(idx_neg)
    cur = 0
    for cur_thr in thr:
        # pos_set = sim[idx_pos]
        num_ta = np.where(sim[idx_pos] > cur_thr)[0].shape[0]
        num_fa = np.where(sim[idx_neg] > cur_thr)[0].shape[0]
        num_tn = np.where(sim[idx_neg] < cur_thr)[0].shape[0]

        FA[cur] = num_fa / num_neg
        TA[cur] = num_ta / num_pos
        acc[cur] = (num_ta + num_tn) / (num_neg + num_pos)
        cur += 1
    order_fa = np.argsort(FA)
    FA = FA[order_fa]
    TA = TA[order_fa]
    auc = 0
    for i in range(FA.shape[0]):
        if i == 0:
            step = FA[i] - 0.00
        else:
            step = FA[i] - FA[i - 1]
        auc += step * TA[i]

    idx_target = np.argmin(np.abs(FA - 0.01))
    tar_far = TA[idx_target]
    print('tar=%f @ far=0.01' % tar_far)
    print('AUC = %f' % auc)
    print('Accuracy = %f' % np.max(acc))
    # import pdb
    # pdb.set_trace()
    return auc, np.max(acc), tar_far
    # min_sim = dist.min()
    # max_sim = dist.max()
    # rng = max_sim - min_sim
    # thresholds = np.arange(dist.min(), dist.max(), rng / 1000)
    # num_thresholds = len(thresholds)
    # threshold_idx = 0
    # tprs = np.zeros((num_thresholds))
    # fprs = np.zeros((num_thresholds))
    # accuracy = np.zeros((num_thresholds))
    # for threshold in thresholds:
    #     tprs[threshold_idx], fprs[threshold_idx], accuracy[threshold_idx] = calculate_accuracy(threshold, dist,actual_issame)
    #     threshold_idx += 1
    # import pdb
    # pdb.set_trace()
    # auc_value = auc(fprs, tprs)
    # max_acc = accuracy.max()
    # return auc_value, max_acc


def cal_acc(predict_label, truth_label):
    predict_label = np.array(predict_label)
    truth_label = np.array(truth_label)
    rp = np.equal(predict_label, truth_label)
    acc = np.sum(rp) / truth_label.shape[-1]

    return acc


def cal_similarity(a_feats, b_feats=None, sim_type='cosine'):
    '''

    :param feats: samples x feature dim
    :return:
    '''

    if b_feats is None:
        b_feats = a_feats
    sim_mat = []
    if sim_type == 'cosine':
        sim_mat = skp.cosine_similarity(X=a_feats, Y=b_feats, dense_output=True)
    return sim_mat


def cal_far(sim, actual_issame, num_interval=1000, save_path='tmp'):
    fars = [0.001, 0.005, 0.01, 0.05, 0.1]
    min_sim = sim.min()
    max_sim = sim.max()
    num = num_interval # (max_sim-min_sim)//1e-2
    thr = np.linspace(min_sim, max_sim, num=num)
    FA = np.zeros(thr.shape, dtype=np.float32)
    TN = np.zeros(thr.shape, dtype=np.float32)
    TA = np.zeros(thr.shape, dtype=np.float32)
    acc = np.zeros(thr.shape, dtype=np.float32)
    idx_pos = np.where(actual_issame == True)[0].tolist()
    idx_neg = np.where(actual_issame == False)[0].tolist()
    num_pos = len(idx_pos)
    num_neg = len(idx_neg)

    cur = 0
    for cur_thr in thr:
        # pos_set = sim[idx_pos]
        num_ta = np.where(sim[idx_pos] > cur_thr)[0].shape[0]
        num_fa = np.where(sim[idx_neg] > cur_thr)[0].shape[0]
        num_tn = np.where(sim[idx_neg] < cur_thr)[0].shape[0]
        #
        FA[cur] = num_fa / num_neg
        TA[cur] = num_ta / num_pos
        acc[cur] = (num_ta + num_tn) / (num_neg + num_pos)
        #tar, far, c_acc = calculate_far_tar(cur_thr, sim, idx_pos, idx_neg, num_neg, num_pos)
        # FA[cur] = far
        # TA[cur] = tar
        # acc[cur] = c_acc
        cur += 1
    order_fa = np.argsort(FA)
    thr = thr[order_fa]
    FA = FA[order_fa]
    TA = TA[order_fa]

    if save_path is not None:
        f = interpolate.interp1d(FA, TA, kind='slinear')
        anchor_fars = np.arange(-4, 0, 0.01)
        anchor_fars = np.power(10, anchor_fars)
        anchor_tars = f(anchor_fars)
        np.save(save_path, anchor_tars)
    # for far in anchor_fars:
    #     # thr = f(far)
    #     thr = 0
    #     # tar, far,_ = calculate_far_tar(thr, sim, idx_pos, idx_neg, num_neg, num_pos)
    #     tar = f(far)
    #     print('far %f tar %f threshold %f' % (far, tar, thr))
    auc = 0
    for i in range(FA.shape[0]):
        if i == 0:
            step = FA[i] - 0.00
        else:
            step = FA[i] - FA[i - 1]
        auc += step * TA[i]
    tars = []
    thrs = []
    for far in fars:
        idx_target = np.argmin(np.abs(FA - far))
        tars.append(TA[idx_target])
        thrs.append(thr[idx_target])
        # tar, _, _ = calculate_far_tar(f(far), sim, idx_pos, idx_neg, num_neg, num_pos)
        # tars.append(tar)
    return fars, tars, thrs, FA, TA, auc, np.max(acc)
def calculate_far_tar(cur_thr, sim, idx_pos, idx_neg, num_neg, num_pos):
    num_ta = np.where(sim[idx_pos] > cur_thr)[0].shape[0]
    num_fa = np.where(sim[idx_neg] > cur_thr)[0].shape[0]
    num_tn = np.where(sim[idx_neg] < cur_thr)[0].shape[0]

    far = num_fa / num_neg
    tar = num_ta / num_pos
    acc = (num_ta + num_tn) / (num_neg + num_pos)
    return tar, far, acc

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))# positive to positive
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))# negtive to positive
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))# negtive to negtive
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))# positive to negtive

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc