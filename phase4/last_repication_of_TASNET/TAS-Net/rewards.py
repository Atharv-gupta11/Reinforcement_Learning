import torch
import random
from utils import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def compute_reward(seq, actions, ignore_far_sim=False, temp_dist_thre=20):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze()
    pick_idxs = torch.nonzero(pick_idxs, as_tuple=False)
    pick_idxs = pick_idxs.squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        reward = reward.to(device)
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_sim = torch.tensor(0.)
        reward_sim = reward_sim.to(device)
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        sim_mat = torch. matmul(normed_seq, normed_seq.t())  # similarity matrix [Eq.4]
        sim_submat = sim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            sim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_sim = sim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
    dist_mat = torch.pow(normed_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat = torch.addmm(input=dist_mat, beta=1, mat1=normed_seq, mat2=normed_seq.t(), alpha=-2)
    dist_mat = dist_mat[:, pick_idxs]
    if len(dist_mat.shape) == 1:
        dist_mat = dist_mat.unsqueeze(dim=1)
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    # --- NEW: Temporal Consistency Penalty ---
    # Prevents "jittery" contextual bandit selections by penalizing
    # excessive transitions (e.g., 0->1->0->1) to enforce contiguous emotion blocks.
    act_sq = _actions.squeeze()
    if n > 1:
        transitions = torch.abs(act_sq[1:] - act_sq[:-1]).sum()
        # reward_temp favors fewer transitions relative to the number of picked frames
        reward_temp = torch.exp(-transitions / (num_picks + 1e-8))
    else:
        reward_temp = torch.tensor(1.0).to(device)

    # Combine rewards: 
    # Downweighting sim/rep slightly to allow the temporal consistency to shape the trajectory
    reward = (reward_sim + reward_rep) * 0.4 + (reward_temp * 0.2)
    return reward

def compute_reward_coff(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False,
                        div_coff=0.5, rep_coff=0.5):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu:
            reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1 - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
    dist_mat = torch.pow(normed_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, normed_seq, normed_seq.t())
    dist_mat = dist_mat[:, pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]

    if _seq.size(1) == 2048:
        reward_rep = torch.exp(-dist_mat.mean()*0.1)
    else:
        reward_rep = torch.exp(-dist_mat.mean())

    # combine the two rewards
    reward = (reward_div * div_coff + reward_rep * rep_coff)
    # print("num_picks {} reward_div {}\t  reward_rep {}\t".format(num_picks, reward_div, reward_rep))
    return reward


def compute_reward_det_coff(seq, actions, det_scores, det_class,  episode,
                            ignore_far_sim=True, temp_dist_thre=20, use_gpu=False,
                            div_coff=50.0, rep_coff=50.0, det_coff =50.0):
    """
    Compute detection reward, diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU

    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu:
            reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
    dist_mat = torch.pow(normed_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, normed_seq, normed_seq.t())
    dist_mat = dist_mat[:, pick_idxs]

    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    det_class = torch.from_numpy(det_class.astype(int))
    det_scores = torch.from_numpy(det_scores)
    sp_det_score = torch.zeros(n) # the score of standard plane detection

    for i in range(n):
        det_cls = det_class[i]
        if det_cls[0] != 3:
            sp_det_score[i] = det_scores[i][det_cls]
        else:
            sp_det_score[i] = -det_scores[i][3]

    pick_scores = sp_det_score[pick_idxs]
    reward_det = torch.sum(pick_scores)/num_picks # (norm_summ_scores)

    # combine the three rewards
    reward = reward_div * div_coff + reward_rep * rep_coff + reward_det  * det_coff
    # reward = (reward_div + reward_rep) * 0.25 + reward_det *0.5
    return reward

