from Module.sparse_net_grey import SparseNet
import torch

root_path = './SLDataSet/20181204/'
# sparse_network = SparseNet(root_path=root_path, batch_size=4, down_k=3)
# sparse_network.load_state_dict(torch.load('./model_sparse_error.pt'), strict=True)
# param_dict = sparse_network.state_dict()

param_dict = torch.load('./model_sparse_error.pt')
param_dict = torch.load('./model_sparse.pt')
# param_dict = torch.load('./model_pattern.pt')

for param_tensor in param_dict:
    print(param_tensor, "\t", param_dict[param_tensor].size(),
          torch.isnan(param_dict[param_tensor]).any().item())

# p_tensor = param_dict['in_layer.0.weight']
# print(p_tensor.reshape(512, 2))
