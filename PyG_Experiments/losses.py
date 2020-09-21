import torch
import torch.nn.functional as F
import numpy as np

DEFAULT_EPS = 1e-10

def group_loss(pred, gt, complex_list, pos_weight, device):
    average_loss_per_complex = torch.tensor(0, dtype= torch.float).to(device)
    unique_complex_list = set(complex_list)
    count = len(unique_complex_list)
    arr_complex_list = np.array(complex_list, np.str)
    for current_complex in unique_complex_list:
        mask = (arr_complex_list == current_complex)
        average_loss_per_complex += F.binary_cross_entropy_with_logits(pred[mask], gt[mask], pos_weight= pos_weight)
        del mask
    del arr_complex_list, unique_complex_list
    average_loss_per_complex /= count
    return average_loss_per_complex

def group_smooth_l1(pred, gt, complex_list, device):
    average_loss_per_complex = torch.tensor(0, dtype=torch.float).to(device)
    unique_complex_list = set(complex_list)
    count = len(unique_complex_list)
    arr_complex_list = np.array(complex_list, np.str)
    for current_complex in unique_complex_list:
        mask = (arr_complex_list == current_complex)
        average_loss_per_complex += F.smooth_l1_loss(pred[mask], gt[mask])
        del mask
    del arr_complex_list, unique_complex_list
    average_loss_per_complex /= count
    return average_loss_per_complex

def group_L1(pred, gt, complex_list, device):
    average_loss_per_complex = torch.tensor(0, dtype=torch.float).to(device)
    unique_complex_list = set(complex_list)
    unique_complex_count = len(unique_complex_list)
    arr_complex_list = np.array(complex_list, np.str)
    #all_sample_count = len(pred)
    for current_complex in unique_complex_list:
        mask = (arr_complex_list == current_complex)
        #weight = float(all_sample_count) / len(pred[mask])
        average_loss_per_complex += F.l1_loss(pred[mask], gt[mask])
        del mask
    del arr_complex_list, unique_complex_list
    average_loss_per_complex /= unique_complex_count
    return average_loss_per_complex

def group_mse(pred, gt, complex_list, device):
    average_loss_per_complex = torch.tensor(0, dtype=torch.float).to(device)
    unique_complex_list = set(complex_list)
    unique_complex_count = len(unique_complex_list)
    arr_complex_list = np.array(complex_list, np.str)
    #all_sample_count = len(pred)
    for current_complex in unique_complex_list:
        mask = (arr_complex_list == current_complex)
        #weight = float(all_sample_count) / len(pred[mask])
        average_loss_per_complex += F.l1_loss(pred[mask], gt[mask], reduction='mean')
        del mask
    del arr_complex_list, unique_complex_list
    average_loss_per_complex /=  unique_complex_count
    return average_loss_per_complex




def group_and_pair_ranking_loss(pred_scores, gt_scores, complex_name, arr_complex_list ):
    mask = (complex_name == arr_complex_list)
    if len(gt_scores[mask]) > 2:
        left_indics, right_indics = get_ranking_pairs(list(gt_scores[mask]))
        pair_num = len(left_indics)
        gt_rank = torch.ones(pair_num, device= pred_scores.device)
        diff_rank = pred_scores[left_indics] - pred_scores[right_indics]
        loss = F.binary_cross_entropy_with_logits(diff_rank, gt_rank, reduction='sum')
        del mask, left_indics, right_indics, gt_rank, diff_rank
        return loss, pair_num
    else:
        del mask
        return torch.tensor(data= 0.0, dtype= torch.float, device= pred_scores.device, requires_grad= True), 0


def get_ranking_pairs(scores):
    """
    compute the ordered pairs whose first doc has a higher value than second one.
    :param scores: given score list of documents for a particular query
    :return: ordered pairs.  List of tuple, like [(1,2), (2,3), (1,3)]
    """
    left_indics = []
    right_indics = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] > scores[j]:
                left_indics.append(i)
                right_indics.append(j)
    return left_indics, right_indics


def split_pairs(order_pairs, true_scores):
    """
    split the pairs into two list, named relevant_doc and irrelevant_doc.
    relevant_doc[i] is prior to irrelevant_doc[i]

    :param order_pairs: ordered pairs of all queries
    :param ture_scores: scores of docs for each query
    :return: relevant_doc and irrelevant_doc
    """
    relevant_doc = []
    irrelevant_doc = []
    doc_idx_base = 0
    query_num = len(order_pairs)
    for i in range(query_num):
        pair_num = len(order_pairs[i])
        docs_num = len(true_scores[i])
        for j in range(pair_num):
            d1, d2 = order_pairs[i][j]
            d1 += doc_idx_base
            d2 += doc_idx_base
            relevant_doc.append(d1)
            irrelevant_doc.append(d2)
        doc_idx_base += docs_num
    return relevant_doc, irrelevant_doc


def rankMSE_loss_function(batch_pred=None, batch_label=None):

    # max_rele_level = torch.max(batch_label)
    # batch_pred = batch_pred * max_rele_level
    return F.mse_loss(batch_pred, batch_label)

def weighted_l1_loss(input, target, weights, reduction = 'mean'):
    ret = torch.abs(input - target) * weights
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def rank_net_loss(pred_scores, gt_scores, complex_list):
    # add regulizer to avoid constant predict scores
    # if torch.var(pred_scores) == 0:
    #     return torch.tensor(data=1.0, dtype= torch.float, device= pred_scores.device, requires_grad= True) + F.l1_loss(pred_scores, gt_scores)
    unique_complex_list = set(complex_list)
    arr_complex_list = np.array(complex_list, np.str)
    pair_count = DEFAULT_EPS
    loss_sum = []
    for current_complex in unique_complex_list:
        current_loss, current_pair_num = group_and_pair_ranking_loss(pred_scores, gt_scores, current_complex, arr_complex_list)
        loss_sum.append(current_loss)
        pair_count = pair_count + current_pair_num
    loss_mean = torch.sum(torch.stack(loss_sum)) / pair_count
    del arr_complex_list, loss_sum
    return loss_mean + F.l1_loss(pred_scores, gt_scores)


def rank_loss(pred_scores, gt_scores, complex_list, loss_name = 'listMLE', combined_loss_name = 'none', epoch= -1):
    unique_complex_list = set(complex_list)
    loss_sum = []
    arr_complex_list = np.array(complex_list, np.str)
    for complex_name in unique_complex_list:
        mask = (complex_name == arr_complex_list)
        if loss_name == 'listNet':
            loss_sum.append(list_top1_CE(pred_scores, gt_scores, mask, combined_loss_name, epoch))
        elif loss_name == 'listMLE':
            loss_sum.append(listMLE(pred_scores, gt_scores, mask, combined_loss_name, epoch))
        else:
            raise Exception('incorrect ranking loss name')
        del mask
    return torch.mean(torch.stack(loss_sum))
    # if combined_loss_name is None:
    #     return mean_rank_loss
    # elif combined_loss_name == 'l1_loss':
    #     return mean_rank_loss + F.l1_loss(pred_scores, gt_scores)
    # elif combined_loss_name == 'smooth_l1_loss':
    #     return mean_rank_loss + F.smooth_l1_loss(pred_scores, gt_scores)
    # elif combined_loss_name == 'mse_loss':
    #     return mean_rank_loss + F.mse_loss(pred_scores, gt_scores)
    # else:
    #     raise Exception('Please give a valid combined_loss_name like: l1_loss, smooth_l1_loss, mse_loss')

def topk_prob(score_list, k):
    y, top_indics = torch.topk(score_list, k, dim= 0)
    ary = torch.arange(k)
    return torch.prod(torch.stack([torch.exp(y[i]) / torch.sum(torch.exp(y[i:])) for i in ary]))

def list_net_loss(y_pred, y_true):
    return torch.sum(-torch.sum(F.softmax(y_true.view(-1, 1), dim=0) * F.log_softmax(y_pred.view(-1, 1), dim=0), dim=1))

def STlist_net_loss(y_pred, y_true, temperature : float = 1.0):
    '''
            The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
            :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
            :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
            :return:
            '''

    unif = torch.rand(y_pred.size(), device=y_pred.device)  # [num_samples_per_query, ranking_size]

    gumbel = -torch.log(-torch.log(unif + DEFAULT_EPS) + DEFAULT_EPS)  # Sample from gumbel distribution
    batch_preds = (y_pred + gumbel) / temperature

    # todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
    # c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
    return torch.sum(-torch.sum(F.softmax(y_true.view(-1, 1), dim=0) * F.log_softmax(batch_preds.view(-1, 1), dim=0), dim=1))


def list_top1_CE(y_pred, y_true, combined_loss_name = None, mask = None):

    preds_smax = torch.softmax(torch.sigmoid(y_pred), dim=0)
    true_smax = torch.softmax(torch.sigmoid(y_true), dim=0)

    # y_true, top_indics = torch.topk(y_true, 200)
    # y_pred = y_pred[top_indics]

    preds_smax = preds_smax + DEFAULT_EPS
    preds_log = torch.log(preds_smax)

    rank_part =-torch.sum(true_smax * preds_log) / len(y_pred)

    if combined_loss_name is None:
        return rank_part
    elif combined_loss_name == 'l1_loss':
        return rank_part #+ 0.4 *  F.l1_loss(y_pred, y_true, reduction= 'mean')
       # return F.l1_loss(y_pred, y_true, reduction='mean')
    elif combined_loss_name == 'smooth_l1_loss':
        return rank_part + F.smooth_l1_loss(y_pred, y_true, reduction= 'mean')
    elif combined_loss_name == 'mse_loss':
        return rank_part + F.mse_loss(y_pred, y_true, reduction= 'mean')
    else:
        raise Exception('Please give a valid combined_loss_name like: l1_loss, smooth_l1_loss, mse_loss')

def listMLE(y_pred, y_true, mask, combined_loss_name, epoch):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[random_indices]
    y_true_shuffled = y_true[random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=0, index=indices)

    max_pred_values, _ = preds_sorted_by_true.max(dim=0, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[0]), dim=0).flip(dims=[0])

    observation_loss = torch.log(cumsums + DEFAULT_EPS) - preds_sorted_by_true_minus_max

    rank_part = torch.sum(observation_loss) / len(y_pred)

    if combined_loss_name is None:
        return rank_part
    elif combined_loss_name == 'l1_loss':
        if epoch < 0:
            return rank_part
        else:
            return rank_part + F.l1_loss(y_pred, y_true, reduction= 'mean')
    elif combined_loss_name == 'smooth_l1_loss':
        return rank_part + F.smooth_l1_loss(y_pred, y_true, reduction= 'mean')
    elif combined_loss_name == 'mse_loss':
        return rank_part + F.mse_loss(y_pred, y_true, reduction= 'mean')
    else:
        raise Exception('Please give a valid combined_loss_name like: l1_loss, smooth_l1_loss, mse_loss')