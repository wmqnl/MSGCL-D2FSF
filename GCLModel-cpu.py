from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class MYGCL(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(MYGCL, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor: #The projection function д in our method is implemented with a two-layer perceptron model.
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,alltraincates): #传入每个类别数据的下标list--》分别取出每个类别嵌入，计算该类别在g1-g1图中的两两相似度和g1-g2图中的两两相似度

        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1)) #投影后的嵌入计算余弦相似度
        between_sim = f(self.sim(z1, z2))
        if len(alltraincates)>0: #train->计算有监督/半监督GCL loss
            allcateemds_z1=[]
            allcateemds_z2=[]
            for x in range(len(alltraincates)):
                currentcateemds_z1=[]
                currentcateemds_z2=[]
                for xx in range(len(alltraincates[x])):
    #                 print(alltraincates[x][xx]) #13
    #                 print(z1[alltraincates[x][xx]]) #tensor([-0.0391, -0.0276, -0.0272, -0.0023, -0.0095,  0.0178, -0.0418, -0.0268,
                    currentcateemds_z1.append(z1[alltraincates[x][xx]].tolist())
                    allcateemds_z1.append(currentcateemds_z1)
                    currentcateemds_z2.append(z2[alltraincates[x][xx]].tolist())
                    allcateemds_z2.append(currentcateemds_z2)

            trainSuper_fz=[]
            for m in range(len(alltraincates)):
    #             for n in range(len(allcateemds_z2)):
                    #先取第0类计算z1-z1，z1-z2同类的所有相似度和
    #             print(m)
                z1_currentcates=torch.tensor(allcateemds_z1[m])
                z2_currentcates=torch.tensor(allcateemds_z2[m])
    #             print(z1_currentcates,z1_currentcates.shape,z2_currentcates,z2_currentcates.shape)
                refl_sim_current = f(self.sim(z1_currentcates, z1_currentcates)) # -refl_sim_current.diag()
                between_sim_current = f(self.sim(z1_currentcates, z2_currentcates))
    #             print(refl_sim_current.sum(1)+between_sim_current.sum(1)-refl_sim_current.diag())
    #             print(Stop)
                trainSuper_fz.append((refl_sim_current.sum(1)+between_sim_current.sum(1)-refl_sim_current.diag()).tolist())
    #         print(trainSuper_fz) #[[383.25445556640625, 384.0289001464844, 383.4620361328125, 385.3174743652344, 383.95562744140625, 387.5706481933594, 382.52294921875, 382.0313415527344, 382.70672607421875, 387.5572814941406, 387.01544189453125, 387.538818359375, 382.74554443359375, 388.0513916015625, 382.9595947265625, 384.1357727050781, 384.7919921875, 382.20233154296875, 384.6425476074219, 385.5051574707031], [383.25445556640625, 384.0289001464844, 383.4620361328125, 385.3174743652344, 383.95562744140625, 387.5706481933594, 382.52294921875, 382.0313415527344, 382.70672607421875, 387.5572814941406, 387.01544189453125, 387.538818359375, 382.74554443359375, 388.0513916015625, 382.9595947265625, 384.1357727050781, 384.7919921875, 382.20233154296875, 384.6425476074219, 385.5051574707031]]
            ConcateTrainSuper_fz=trainSuper_fz[0]
#             print(len(trainSuper_fz)) #7
            for n in range(len(trainSuper_fz)):
                if n!=0:
                    ConcateTrainSuper_fz=ConcateTrainSuper_fz+trainSuper_fz[n] #0+1+2

            return -torch.log( torch.tensor(ConcateTrainSuper_fz)/(refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
                 
        else:
            return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,alltraincates, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2,alltraincates)
            l2 = self.semi_loss(h2, h1,alltraincates)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


