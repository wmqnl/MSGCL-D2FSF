import torch
import torch.nn as nn

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
#           
            return torch.mean(seq, 1) #表示图的嵌入/表示？
        else:
            msk = torch.unsqueeze(msk, -1) #扩展一个维度  (有掩码的节点求和，取均值？
            return torch.sum(seq * msk, 1) / torch.sum(msk)

