import types
import torch

def read_txt(path):
    """
    read path and save the content into list
    """

    list = []
    with open(path, 'r') as f:
        for line in f:
            list.append(line.strip())

    return list

def write_txt(list, path):
    """
    write list into path
    """

    f = open(path,'w')
    for item in list:
        f.write(str(item))
        f.write('\n')
    f.close()

def load_pt(file_path):
    data = torch.load(file_path)
    node_features = data.x.numpy()
    label = data.y.numpy()
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()
    interface_count = data.interface_count.numpy()
    interface_pos = data.interface_pos.numpy()
    complex_name = data.complex_name
    decoy_name = data.decoy_name
    del data

if __name__ == '__main__':
    load_pt(r'/home/fei/Research/processed_all_data/classification/1BVN.complex.8199.pdb.pt')