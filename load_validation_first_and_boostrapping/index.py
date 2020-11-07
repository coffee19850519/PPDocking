import torch
from torch_geometric.data import Data,DataLoader


# reindexes the residue_num globally
# 1 batch = 1 protein
# can identify which res_num belongs to which protein/batch based on the batch variable
# note: this function assumes sequental nature of res_num 
# TODO: if data is not sequential per protein: per protein sort, save sorted indeces, run below, then revert sort
def global_index_residue(b_res_num):

    res_index = 0
    temp_res = None
    for x in range(b_res_num.shape[0]):

        current_res = b_res_num[x].clone()

        # at start set temp_res to first residue
        if not torch.is_tensor(temp_res):
            temp_res = b_res_num[x]

        if not torch.eq(temp_res, current_res):
            res_index += 1
            temp_res = current_res

        b_res_num[x] = res_index

    return b_res_num



data_list = []
for x in range(48):

    edge_index = torch.tensor([[0, 1],[1, 0],[1, 2],[2, 1],[1,3],[3,1],[4,2],[2,4],[5,1],[1,5]], dtype=torch.long)
    x = torch.randn(12, 16)
    res_num = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1], [2], [2], [2], [2]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), res_num=res_num)
    data_list.append(data)

loader = DataLoader(data_list,batch_size=4)
batch = next(iter(loader))

batch_batch = batch.batch
b_edge_index = batch.edge_index
b_x = batch.x
b_res_num = batch.res_num

b_res_num = global_index_residue(b_res_num)
