import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader # torch.utils.data.dataloader
# from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,accuracy_score
from utils2 import load_data
from torch_geometric.data import Data #pgl版本的
from readout import AvgReadout #导入readout库
import os, gc, sys
from print_log import Logger
import pickle
import networkx as nx
import dgl
from dgl import DGLGraph
from torch_geometric.utils import degree,to_undirected, dropout_adj
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv #    #base_models[name] : <class 'torch_geometric.nn.conv.gcn_conv.GCNConv'>
# from tables.table import Cols
import argparse
# from spparam import SimpleParam
from GCLModel import Encoder, MYGCL#无监督图对比学习模块
from GCLutils import get_base_model, get_activation
from crossattention import CrossAttention
# from projectLayer import ProjectLayer #导入MLP
from sklearn.metrics import precision_recall_fscore_support #P,R,F1新指标
from sklearn.metrics import accuracy_score
from copy import deepcopy
import time
starttime=time.time() 
# 注意：10000多的loss---》将Adam优化器换成SGD优化器，则loss变为10左右
# 11.5--21:02:agnews-difformerHeatKnerel500
###d第二阶段调试效果： 渐进式调试（局部技术/关键技术）、）
#   （1）g1图节点嵌入（评判标准：仍使用HeatKernel构图+sentence-transformer网站的节点嵌入方式构造g1：测试~~和当前的g1train，val，test比较）---》
#     （2）g1图构造方式--》（標準：使用cos/欧式距离等构图：测试~~当前的g1trainvaltest比较）
#    ====c》“第一阶段调整”
#     （3）特征重要性和边级重要性是否要“计算相对重要性”-进一步规范化得分：+优化的（1）（2）基础上“别计算相对重要性”--再图融合，测试~~比较（1）（2）的acc
#    =============d“第二阶段图融合阶段调整”
# 查看dgl版的api--对照写自己的代码
# Training PPI Dataset with DGL built-in GATConv module.
# Downloading C:\Users\Administrator\.dgl\ppi.zip from https://data.dgl.ai/dataset/ppi.zip...
# Extracting file to C:\Users\Administrator\.dgl\ppi_4b14ad03
# Training...
# Epoch 00000 | Loss 0.7005 |
# Epoch 00001 | Loss 0.5376 |,...
# vali graphs:6514 (2 graphs),test graph:5524 (2 graphs)
#按照IPM 2022z中的degree重要性方式推导出g1图中每条边的权重：  计算节点degree中心性方式的---导入torch，geometric中的degree函数： from torch_geometric.utils import degree
# def degree_drop_weights(edge_index):
#     #import ipdb;ipdb.set_trace()
#     edge_index_ = to_undirected(edge_index)
#     deg = degree(edge_index_[1]) #计算节点度
#     deg_col = deg[edge_index[1]].to(torch.float32)
#     s_col = torch.log(deg_col)
#     weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
#     return weights  #计算边的权重
# deg = degree(edge_index_[1])
# deg_col = deg[edge_index[1]].to(torch.float32)
#边权重规范化：注意换一种方式规范化（不和IPM相同）--换一种与IPM中不同的方式来规范化权重  edge_weights = edge_weights / edge_weights.mean() * p
# 是否需要计算删除边的权重： support = drop_edge_weighted(edge_index_self, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=cfg.threshold)
logdir = "log/"
# dataset='agnews'
# sys.stdout = Logger(logdir + "{}-totallossbp.log".format(dataset))  #写入日志文件中  
# sys.stdout = Logger(logdir + "{}-totallossxiuzheng.log".format(dataset))  #写入日志文件中  
##z注意参考STCKA整个模型代码模式：将图融合代码（即，模块-函数，子模块-函数等）写在DFGAT模型中+"通过forward函数按流程调用"++不同的图融合模块及其包含的子模块
#    ---------------------->(1) forward中可以更新各种模型參數：包括全连接参数，特征映射参数，MLP（project参数等）--通过BP算法自动更新参数
#    ----------------------->(2)m模型可以自动学习需要的参数，如损失系数eta等
#         print(stop)
#第二种方案：“图融合+嵌入到GAT模型训练和learning过程“，  每个epoch都要进行图融合=》得到新的g3图，同时更新参数----》进一步“使用更新参数的GAT模型”train/val等
# ====>j接着，新的epoch“再次图融合，更新融合后的g3图，并更新参数”---=--》接着，train/val，。。。。重复这个过程
# class STCK_Atten(nn.Module): 
class DGF_MSGCL(nn.Module): #模型中构造函数初始化 --GAT模型传入了4个参数--包括FeatureImportance_Compute函数和forward函数
#     def __init__(self, in_size, hid_size, out_size, heads,GFusion): #attention中的multi-head注意力
    def __init__(self, in_size, hid_size, out_size, heads): #attention中的multi-head注意力
        super().__init__()
        self.gat_layers = nn.ModuleList()
#         self.GFusion=GFusion #bool标记
#         self.read = AvgReadout() #readout函数读取图的表示
        dim=384 #隐层维度，256*4
        self.crossatt=CrossAttention(dim) #self封装crossattention类和投影类到“当前DFGAT模型中“，“随着DFGAT模型BP更新内嵌模块的参数”
#         self.projectLayer =ProjectLayer()
        dtemp = 32
        self.W = nn.Linear(dim, dtemp) 
#         self.W = nn.Linear(256*out_size, dtemp) # self.W和self.w不是weight参数？？ c = self.w(F.tanh(self.W(c))) --C-CS注意力对应的权重参数
        self.w = nn.Linear(dtemp, 1, bias=False) #不使用偏置b， 300维-》32维-》1维【神经元数量的变化】， nn.Linear()函数用于设置网络中的全连接层，注意全连接层的输入和输出都是二维张量，batch_size,sie
        # three-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu)
        )
        self.gat_layers.append( #2层GAT层
            dglnn.GATConv( #GATConv中的方法--gatconv.py中的dgl.nn.pytorch.conv库中的函数。
                hid_size * heads[0],
                hid_size,
                heads[1], #隐层注意力头数
                residual=True,
                activation=F.elu,
            )
        )
        print("self.gat_layers: ",self.gat_layers)
        print( dglnn.GATConv( #GATConv中的方法--gatconv.py中的dgl.nn.pytorch.conv库中的函数。
                hid_size * heads[0],
                hid_size,
                heads[1], #隐层注意力头数
                residual=True,
                activation=F.elu,
            ))
#         print(stop)
        self.gat_layers.append(
            dglnn.GATConv(  
                hid_size * heads[1],
                out_size,
                heads[2],
                residual=True,
                activation=None,
            )
        )
    def forward(self, g, inputs): #模型前向传播函数，，不断更新节点表示
        h = inputs.float()
#         h=inputs
#         print("h:",h)
#         h: tensor([[-0.0040, -0.1768,  0.2434,  ..., -0.1804, -0.3630,  0.2488],
#         [ 0.0735, -0.1637, -0.0454,  ..., -0.1746, -0.1356,  0.1198],
#         [-0.0877, -0.0848, -0.0250,  ..., -0.1961,  0.2767, -0.1000],
#         ...,
#         [-0.0041,  0.3287, -0.0255,  ...,  0.1661,  0.0480, -0.3080],
#         [-0.0416, -0.1997,  0.3488,  ..., -0.0242, -0.1677,  0.3140],
#         [-0.0581,  0.0383,  0.2592,  ..., -0.3347, -0.2076,  0.0238]],
#        dtype=torch.float64)
#         print(h.shape) #torch.Size([80, 384])-feature
#         print("g: ",g)
#         g:  Graph(num_nodes=80, num_edges=2750,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={})
#         print(g.shape)
#         print("--h: ",h)
#         print("--h shape: ",h.shape)
        for i, layer in enumerate(self.gat_layers):
#             h = layer(g, h)
            h, edge_attention = layer(g, h) 
            if i == 2:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
#         print("h: ",h)
#         print("h shape: ",h.shape) #每个epoch输出10个--h  shape:  torch.Size([3144, 121])--121类,h shape:  torch.Size([4602, 121]),...h shape:  torch.Size([6184, 121])--10次
#         h:  tensor([[ 0.0264, -0.1377,  0.1005,  ...,  0.0917,  0.1397, -0.0576],
#         [ 0.0873, -0.0937, -0.0662,  ..., -0.1017,  0.0064, -0.0263],
#         [-0.0341,  0.0351, -0.0360,  ..., -0.0804, -0.0005,  0.1689],
#         ...,
#         [ 0.0666, -0.1268,  0.0167,  ..., -0.0565, -0.0356,  0.0025],
#         [ 0.0409,  0.0344, -0.1442,  ..., -0.0695,  0.2048, -0.0963],
#         [-0.0505,  0.0677,  0.2055,  ..., -0.0293, -0.3340,  0.2958]],
#        grad_fn=<MeanBackward1>)
#         print(stop)
        return h

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
#***y一定要从多角度分析出现问题的原因，根据报错-追溯报错附近代码---从“不同角度入手”调试代码、分析代码、排查原因
def encode_onehot(dlabels):
    labelsls=dlabels.tolist()
    newlabels=[]
    for i in range(len(labelsls)):
        val=str(labelsls[i])
        newlabels.append(val)
    print(newlabels,len(newlabels)) #['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '
    newlabelsarr=np.array(newlabels)
    print(newlabelsarr,newlabelsarr.shape) # '0' '0' '0' '0' '0' '0' '0' '0' '0' '0'] (1000,)
    classes = set(newlabelsarr)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, newlabelsarr)),
                             dtype=np.int32)
    return labels_onehot
def evaluate(g, features, labels,onehotlabs, model): #1个训练图，调用这个？
    model.eval()
    correct = 0
    total = 0
#     with torch.no_grad(): # 测试时不需要梯度下降？   output = model(input_features_train, input_adj_train)#调用HGCN模型时传入所有节点的特征和adj
    print("g: ",g)
    print("features: ",features)
    print(features.shape) #torch.Size([7721, 300]
    output = model(g, features) 
#     g:  Graph(num_nodes=7721, num_edges=1975027,
#       ndata_schemes={'feat': Scheme(shape=(300,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={})
# features:  tensor([[-0.0301, -0.0805, -0.0195,  ...,  0.0281,  0.0275,  0.0316],
#         [-0.0357, -0.0424,  0.0027,  ..., -0.0146,  0.0573,  0.0159],
    preds_probs = output.cpu().detach().numpy()
    preds = deepcopy(preds_probs)
    preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
    preds[np.where(preds < 1)] = 0.0
    print(onehotlabs.shape,preds.shape) #(1000, 2) (1000, 2)
    print(onehotlabs,preds) 
    
    for i in range(len(preds)):
        max_value=max(preds[i])
        for j in range(len(preds[i])):
            if max_value==preds[i][j]:
                preds[i][j]=1
            else:
                preds[i][j]=0  
    [precision, recall, F1, support] = \
        precision_recall_fscore_support(onehotlabs, preds, average='macro') #Macros calculate metrics for each label, and find their unweighted mean, where the labeled
# imbalance is not considered:
    Acc = accuracy_score(onehotlabs, preds) * 100
   
    return Acc,precision, recall, F1
#         print(stop)
#         return score


def evaluate_in_batches(dataloader, onehotlabs,device, model): #评估时只传入testdataloader
    total_score = 0
    for batch_id, batched_graph in enumerate(dataloader):
      
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata["feat"]
        labels = batched_graph.ndata["label"]
#         print("batch_id: ",batch_id) #batch_id:  0
#         print("batched_graph: ",batched_graph)
#         print("features: ",features)
#         print("labels: ",labels)
#         print("features shape: ",features.shape)
#         print("labels.shape: ",labels.shape)
#         print(stop)
# ValueError: Classification metrics can't handle a mix of multiclass and multilabel-indicator targets

        score,precision, recall, F1 = evaluate(batched_graph, features, labels,onehotlabs, model) #单个图评估调用这个

#         total_score += score #所有训练图得分求和
        print("total_score: ",score)
#     print("total_score / (batch_id + 1) : ",total_score / (batch_id + 1) )
    return score,precision, recall, F1 # return average score

# 如果你的是多分类（or多标签2分类），你可以将你的损失函数改为BCEWithLogitsLoss
# pred:  [[1 0 0 1]
#  [0 0 1 0]
#  [0 0 0 0]
#  ...
#  [0 0 1 1]
#  [1 0 1 0]
#  [1 0 1 0]]---同一个train，valid，test图-total_score / (batch_id + 1) :  0.35412937867553057
# CrossEntropyLoss:pred:  [[1 0 0 1]
#  [1 1 1 1]
#  [1 0 1 1]
#  ...
#  [1 1 1 1]
#  [1 1 1 1]
#  [1 0 0 1]]---封装不需要输入 [ 0 0 0 0 1]，只需要输入 4 就行---函数内部会自己处理成 one hot 格式
def train(train_dataloader, val_dataloader, device, model):
    # define loss function and optimizer
#     loss_fcn = nn.BCEWithLogitsLoss() #ValueError: Target size (torch.Size([8040])) must be the same as input size (torch.Size([8040, 4]))

#     loss_fcn=nn.MSELoss()
    loss_fcn=nn.CrossEntropyLoss()
#需要把HGAT数据集中的networkx格式的graph--》dgl格式的graph（使数据集中的格式和dataset[0]完全保持一致）
#     loss_fcn=nn.NLLLoss()
    print("loss_fcn: ",loss_fcn)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0) #按GAT论文：GRAPH ATTENTION NETWORKS“来设置迪比方法GAT的参数

#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.003)
    print("optimizer: ",optimizer)
    # training loop
    for epoch in range(500):
#     for epoch in range(1000):  #E3000-正式的train、val与test可以设置更大些的epoch，如2000/3000+GPU上---更高的acc
        model.train()
        logits = []
        total_loss = 0
        # mini-batch loop
        for batch_id, batched_graph in enumerate(train_dataloader): #KeyError: 0
            batched_graph = batched_graph.to(device)
#     g.node[ind]['type'] = cate #根据下标给每篇短文本的节点添加type属性值，到图g中
#             labels = batched_graph.ndata[ind]["type"].float()
#             print("labels:",labels)
#             print("batched_graph: ",batched_graph)
#             batched_graph:  Graph(num_nodes=80, num_edges=2750,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={})
            features = batched_graph.ndata["feat"]#RuntimeError: expected scalar type Double but found Float
#             features = batched_graph.ndata["feat"].float64()
            labels = batched_graph.ndata["label"].long() #RuntimeError: expected scalar type Long but found Int
#             labels = batched_graph.ndata["label"].float() #RuntimeError: expected scalar type Long but found Float
#             print("batched_graph.ndata['feat']: ",batched_graph.ndata['feat'])
#             print("batched_graph.ndata['label']: ",batched_graph.ndata['label'])
#             print("features: ",features)
#             print("labels:",labels)
# labels: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
#         2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1,
#         1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2,
#         2, 2, 2, 3, 3, 3, 3, 3])
            logits = model(batched_graph, features)
#             print("batch_id: ",batch_id) #batch_id:  0
#             print("batched_graph: ",batched_graph)
#             print("features: ",features)
#             print("labels: ",labels)
#             print("features shape: ",features.shape)
#             print("labels.shape: ",labels.shape)
#             print("logits: ",logits)
#             print("logits shape:",logits.shape) #logits shape: torch.Size([8040, 4])
#             features shape:  torch.Size([8040, 384])
# labels.shape:  torch.Size([8040, 4])
# logits:  tensor([[ 145.7702,  -77.5904,  237.0179,  -73.0088],
#         [ 109.2722,  170.6721,  326.5601,  -55.5018],
#         [-222.6138,  197.2946, 1911.9058, -372.7206],
#         ...,
#         [-180.2477,  191.7997,  347.0723,  -10.2706],
#         [  38.9388,  228.4832,  233.6712, -115.7943],
#         [-129.3462,  293.4632,  259.6143,   92.2166]], grad_fn=<MeanBackward1>)
#             loss = loss_fcn(logits[train_mask], labels[train_mask])
            loss = loss_fcn(logits, labels) #损失函数  loss_fcn=nn.CrossEntropyLoss()
            print("loss: ",loss)
            print("loss.item(): ",loss.item())
            print(" total_loss: ", total_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            "Epoch {:05d} | Loss {:.4f} |".format(
                epoch, total_loss / (batch_id + 1)
            )
        )

        if (epoch + 1) % 5 == 0: #每5轮评估一次
            Valavg_score,precision,recall,F1 = evaluate_in_batches( #评估时，test的2个图求平均分
                val_dataloader, valonehotlabs,device, model
            )  # evaluate F1-score instead of loss
            print(
                "                       Test Acc.  {:.4f}, precision.  {:.4f}, recall.  {:.4f}, F1.  {:.4f}  ".format(
                     Valavg_score,precision, recall, F1
                ))
            print("Testing...")
            test_score,precision,recall,F1 = evaluate_in_batches(
                test_dataloader, testonehotlabs,device, model
            )
            print(
                "                       Test Acc.  {:.4f}, precision.  {:.4f}, recall.  {:.4f}, F1.  {:.4f}  ".format(
                    test_score,precision, recall, F1
                ))

def Construct_Graph3(g2_edgesweight,edgeatts,edgeindex,edgeindex_values,features,txtsim3): #除传入的三个参数外，其它参数若能调用到，就可以正常运行
#可以使用一些trick，如F.tanh,F.softmax等非线性激活函数，全连接层线性变换等,同时对g1和g2图进行变换-----提升模型的鲁棒性和性能.....
#去掉train标记，变成公共的变量，（对train/val/test不作明显的区分）
#         print(edgeatts[0],edgeatts[0].shape)
#         print(edgeatts[1],edgeatts[1].shape)
#         print(NodeIms[0],len(NodeIms[0])) #140 0.00022920857905885396, 0.00011464623589044311, 7.720650795331528e-05, 
#         print(NodeIms[1],len(NodeIms[1]) ) #140 [0.00023115209265783126, 0.00013254876746676312, 8.803885067634672e-05
#         规范化边级注意力和节点级注意力
#         print(edgeatts)
#           [tensor([[0.0286],
#         [0.0419],
#         [0.0346],
#         ...,
#         [0.0339],
#         [0.0379],
#         [0.0340]], grad_fn=<MeanBackward1>), tensor([[0.0284],    tensor([[0.0284]
#         [0.0249],
#         [1.0000]], grad_fn=<MeanBackward1>)] 先让g1，g2图中每条边权重一一对应
        #组合后按行规范化，每行的和为1.0--概率形式
#         print(stop)
#         edgeatts[0]=F.tanh(edgeatts[0])
#         print(edgeatts[0],edgeatts[0].shape)
#         tensor([[0.0287],
#         [0.0330]], grad_fn=<TanhBackward0>) torch.Size([1312, 1])
#         edgeatts[1]=F.tanh(edgeatts[1])
#         print(edgeatts[1],edgeatts[1].shape)
#           [0.7616]], grad_fn=<TanhBackward0>) torch.Size([947, 1])
#         print(stop)
#   
# 规范化,edgeatts[1]:
# tensor([[0.7616],
#         [0.7616],...grad_fn=<TanhBackward0>) torch.Size([140, 1])
# tensor([[1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],rad_fn=<MeanBackward1>) torch.Size([140, 1])
#         print(Stop)
        g1_edgeatt=edgeatts[0].tolist()
        g2_edgeatt=edgeatts[1].tolist()
#         print(g1_edgeatt) #每次计算的重要性值稍微有差异，  [[0.02495366707444191], [0.0220850370824337], [0.017476219683885574], [0.028818974271416664],       #         print(stop)---注意，传入的edge_index和edge_index_values为train图的
# dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got 1000 and 80 instead.
        edgescores2=dict()
        count=0
        for key,value in g2_edgesweight.items():
#             print("g2_edgeatt[count][0]: ",g2_edgeatt[count][0]) #g2_edgeatt[count][0]:  0.02317703329026699
            edgescores2[key]=g2_edgeatt[count][0] #
#             print(edgeweights2)  #g2_edgeatt[count][0]:  0.023050539195537567
#             print(key,value) #(0, 0) 1.0
            count=count+1
#         print(edgeweights2,len(edgeweights2)) #947 {'(0, 0)': 1.0, '(0, 3)': 0.9400544762611389, '(0, 4)': 0.7392165660858154, '(0, 5)': 0.775178849697113, '(0, 6)': 0.8100672960281372,
# ----c一般情况下，g2的边数<g1的边数---两种方案：（1）取并集： 保留g1中已存在的所有边，，取g2中的各边进行融合（g1中有g2中没有的则保留）（2）取交集：只取g1，g2中都存在的边，舍弃g1存在g2不存在的边
#开始进行图融合，先分别实现边融合和特征融合，再汇总
 #边融合：取出train_g1和train_g2中各自的边和对应的权重weight1，weight2------使用edgeatts中各自的注意得分加权求和 （weight1i*score1i+weight2i*score2==>新的边权重weight3i(g3边的权重）
 #（1）方案一：取g1和g2中的边的并集  以统一的存储方式存储g1和g2中的边下标+值---》以train_g2:trainedgeweights的"key:value"为基准存储+边各自的重要性得分--》更新该边的权重
#        方案二：1取g1，g2的交集 再融合（本质为g2边---》2进一步根据weight大小再过滤：最终g3边数<=g2<=g1
        edgeindex1= edgeindex.tolist() # edgeindex_:无向图
#         print(len(edgeindex1[0]),len(edgeindex1[1])) #2592 2592 注意括号
        edgeindex_values1= edgeindex_values.tolist() 
#          nxcontents=[]
        edgeweights1=dict() #g1图
        for i in range(len(edgeindex1[0])):
            templs=[]
            templs.append(edgeindex1[0][i]) #对应行
            templs.append(edgeindex1[1][i]) #对应列
            temptuple=tuple(templs) 
#             print(temptuple) #(0, 0)
#             nxcontents.append(temptuple)
            edgeweights1[str(temptuple)]=edgeindex_values1[i] #g1:edgeindex1:weight1
        #开始图融合，g1+g2-》g3  >"c整体上对图g1和图g2上对应的节点进行融合”，如f3=b1*f1+b1'*f2 (f3为融合后的节点的新特征）
#                             边融合类似：g1图上边的权重weightg1i*score1i+g2图上边的权重weight2i*score2i
#         print(len(edgeweights1)) #edgeweights1, 1312
      
        edgescores1=dict()
        count1=0
        for key,value in edgeweights1.items():
#             print("g1_edgeatt[count1][0]: ",g1_edgeatt[count1][0]) #g2_edgeatt[count][0]:  0.02317703329026699
            edgescores1[key]=g1_edgeatt[count1][0] #
#             print(edgeweights1)  #g2_edgeatt[count][0]:  0.023050539195537567
#             print(key,value) #(0, 0) 1.0
            count1=count1+1
        newedge_weights=dict() #g1+g2->g3（边权重）
        w_greats=[] #>=0.6c 的权重记录下来
#         for kk,vv in edgeweights1.items():  #以g1为准开始融合
        #查找g1中依次包含g2的边对应的下标
        g1updated_Edgescores=dict()
        g2updated_Edgescores=dict() #更新key-value出现顺序
        Normg1g2_Edgescores=[]
        g1UpdateEdScoresls=[]
        g2UpdateEdScoresls=[]
        g1g2_keys=[]
        for kk,vv in edgescores1.items(): 
            if kk in edgescores2: #当前key位于g2
                g1g2_keys.append(kk)
#             print(temptuple) #(0, 0)
#             nxcontents.append(temptuple)
                edgeweights1[str(temptuple)]=edgeindex_values1[i] 
                g1updated_Edgescores[kk]=edgescores1[kk]
                g2updated_Edgescores[kk]=edgescores2[kk]
                g1UpdateEdScoresls.append(edgescores1[kk])
                g2UpdateEdScoresls.append(edgescores2[kk])
        Normg1g2_Edgescores.append(g1UpdateEdScoresls)
        Normg1g2_Edgescores.append(g2UpdateEdScoresls)
#         print(Normg1g2_Edgescores) #[[0.028525909408926964, 0.026272594928741455, 0.031174220144748688, 0.030320892110466957, 0.02938689850270748, 0.027738478034734726, 
        Normg1g2_Edgescores=torch.Tensor(Normg1g2_Edgescores)
#         print(Normg1g2_Edgescores,Normg1g2_Edgescores.shape)
#         tensor([[0.0286, 0.0263, 0.0312,  ..., 0.0257, 0.0334, 0.0334],
#         [0.0283, 0.0262, 0.0311,  ..., 0.0309, 0.0309, 0.0292]]) torch.Size([2, 947])
        Normg1g2_Edgescores=F.softmax(Normg1g2_Edgescores,dim=0) #按列规范化，对应位置元素和为1
#         print(Normg1g2_Edgescores,Normg1g2_Edgescores.shape)
#     tensor([[0.0286, 0.0263, 0.0313,  ..., 0.0256, 0.0332, 0.0332],
#         [0.0284, 0.0260, 0.0310,  ..., 0.0272, 0.0273, 0.0260]]) torch.Size([2, 947])
#规范化后的边级得分/重要性得分 tensor([[0.5000, 0.5001, 0.5001,  ..., 0.4996, 0.5015, 0.5018],
#         [0.5000, 0.4999, 0.4999,  ..., 0.5004, 0.4985, 0.4982]]) torch.Size([2, 947])
        NormallEdgeScoresls=Normg1g2_Edgescores.tolist()
#
#         print(g1g2_keys,len(g1g2_keys)) #947  ['(0, 0)', '(0, 3)', '(0, 4)', '(0, 5)', '(0, 6)', '(0, 8)', '(0, 10)', '(0, 11)', '(0
#         print(Stop)
        g1Norm_EdgeScores=dict()
        g2Norm_EdgeScores=dict()
        #重新更新g1updated_Edgescores，g2updated_Edgescores,取g1updated_Edgescores的key值+NormNormallEdgeScoresls的value合并成新的dict
        for x in range(len(NormallEdgeScoresls[0])):
            g1Norm_EdgeScores[g1g2_keys[x]]=NormallEdgeScoresls[0][x]
            g2Norm_EdgeScores[g1g2_keys[x]]=NormallEdgeScoresls[1][x]
#         print(len(g1Norm_EdgeScores),g1Norm_EdgeScores)
#         print(len(g2Norm_EdgeScores),g2Norm_EdgeScores)

# 
#         print(Stop)
        g3_newedgeweights=dict()
        for kk,vv in g2_edgesweight.items():  #边的下标-边的权重-------边的下标--边的得分====》两个下标一一对应
#             newweight=0.0
#             if kk in edgeweights2: #该边存在于train_g2: g1，g2都有
#                 print(vv,edgescores1[kk])
#                 print(edgeweights2[kk],edgescores2[kk])
# torch.Size([6408, 1])
# torch.Size([3823, 1])
#  print(edgeweights2,len(edgeweights2)) #947 {'(0, 0)': 1.0, '(0, 3)': 0.9400544762611389, '(0, 4)': 0.7392165660858154, '
            newweight=edgeweights1[kk]*g1Norm_EdgeScores[kk]+g2_edgesweight[kk]*g2Norm_EdgeScores[kk] #g1图的当前边的weight*g1当前边得分+g2图当前边weight*g2当前边得分
#                 print("newweight: ",newweight) #newweight:  0.04768973961472511 newweight:  1.0635367184877396
#0-0：  1.0*0.0249846912920475+1.0*0.022705048322677612----newweight:  0.04768973961472511
#                 print(stop)
#             print(kk,newweight) #(0, 0) 0.99999982114646
#             print(edgeweights1[kk],g1Norm_EdgeScores[kk],edgeweights2[kk],g2Norm_EdgeScores[kk])#0.9999996423721313 0.500142514705658 1.0 0.49985748529434204
#             print(Stop)
#             else:
#                 newweight=vv #原始g1权重
#             newedge_weights[kk]=newweight
            if newweight>=txtsim3: #textsim3，才添加此时的边和weight---》g3中
                g3_newedgeweights[kk]=newweight
#         print(len(g3_newedgeweights),g3_newedgeweights) #0.5: 771 0.6: 223 {'(0, 0)': 0.9999997912980305, '(0, 
        for j in range(features.shape[0]): #g = dgl.add_self_loop(g)` 0-6919, 加上所有对角线的元素，如0-0,1-1,2-2：确保所有节点都在新的dgl图g2中
#         for k in range(len(new_rows)):
#             if new_rows[k]!=j and new_cols[k]!=j: #0-0，相同下标的值已在行标new_rows list和new_cols list中同时存在
            c_temp1=[]
            c_temp1.append(j)
            c_temp1.append(j)
            c_temp1tuple=tuple(c_temp1)
#         print(c_temp1) #[30, 30] aaa
            if str(c_temp1tuple) not in g3_newedgeweights.keys():
#                 print(c_temp1tuple)
                g3_newedgeweights[str(c_temp1tuple)]=1.0  #自连接，0-0，
#         print(features.shape[0],len(g3_newedgeweights),g3_newedgeweights) #40 771 {'(0, 0)': 0.9999998509347989, '(0, 3)': 0.5701732449269059, '(0, 6)': 0.5396937878284511, '(0, 8)':
#             print("aaa")
#         print(Stop)
        
        #三阶段，利用生成的Graph_g3边和特征以及g1，g2共享的label==》构造Graph_g3图
        nxcontents3=[]
#         edges_weights=dict()
        for k3,v3 in g3_newedgeweights.items():
#             print(eval(k3)) #(0, 0)
#             print(Stop)
            nxcontents3.append(eval(k3)) #str->tuple
        Graphg3=nx.DiGraph(nxcontents3) #创建networkx图
        Graph_g3=dgl.from_networkx(Graphg3).to(device)
        Graph_g3.ndata["feat"]= features  #仍先默认为g1,g2初始特征
        # print("Graph_g3: ",Graph_g3)
#         print(Stop)
#         Graph_g3:  Graph(num_nodes=40, num_edges=771,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64)}
#       edata_schemes={})
# Graph_g3:  Graph(num_nodes=80, num_edges=2592,
#       ndata_schemes={}
#       edata_schemes={})
#         print(Stop)
#         W3 = nn.Linear(g3_features.shape[1],in_size) #初始化一个Linear函数W3
#         g3trans_features=W3(g3_features) #传入具体的实例，80*1024的特征--》变换为80*300的特征
#         print(g3trans_features.shape,g3trans_features) #torch.Size([80, 300])  Linear(in_features=1024, out_features=300, bias=True)
#         Graph_g3.ndata["feat"]= g3trans_features #g3图的features应该变换为300维，和g1，g2的初始特征维度相同
#         return Graph_g3,g3_features,g3trans_features,features  #传入原特征， 返回隐层的g3特征还是映射后的特征？
        return Graph_g3,features
#未涉及原图g1上的边weight值:只需传入g1图中边的下标（行标和列标），当前图的短文本-短文本相似度阈值，当前图节点数量
def Construct_Graph2Edges(edge_index,textsim,numnodes): #评估时只传入testdataloader
    edge_index_ = to_undirected(edge_index) #将图转为无向图（ 边的方向是无向的）
    deg = degree(edge_index_[1]) #计算节点中心性， 计算节点度------根据给定的边的“出度”-》计算节点度---weight?
    degcol = deg[edge_index[1]].to(torch.float32) #计算边的中心性，  边的列标  1/2(rJ边的入度+边的出度）----》节点的度===>推导出边的度 （节点中心性----》推导边的中心性）#---c权重--多少条边==》多少个weight---换一种字母表示形式
    scol = torch.log(degcol) #loge为底（即，IN为底）--取对数的边权重  #注意，搜素文献资料------"替换/更换"一种规范化权重的方式===》重新对"logweight"-规范化。。。【使用Min-Max数据归一化方法，而不是"max-mean"归一化。
#   公式为y=(x-min(x))/(max(x)-min(x))---新序列yi=[y1,y2, ...,yn],[0,1]无量纲，一般数据先进行规范化处理，Min-Max规范化是对原始数据线性变换到0-1之间。
    weights = (scol - scol.min()) / (scol.max() - scol.min())
#     weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean()) #max-mean规范化的边权重--注意“规范化的值不在[0,1]之间。。。
    print(scol.max(),scol.mean(), scol.min()) #tensor(3.6889) tensor(3.5035) tensor(2.8332) 所有数据中最大最小值:tensor(4.1109) tensor(2.3026)
#     print("weights: ",weights) #weights:  tensor([0.7666, 0.8439, 0.9721,  ..., 0.8066, 0.9117, 0.8193])
    #根据节点重要性-经过max-mean规范化的边的权重： weights:  tensor([0.7777, 0.5201, 0.0929,  ..., 0.6444, 0.2942, 0.6021])
#     g1图上原来的边的权重： tensor([1.0000, 0.6097, 0.6346,  ..., 0.6499, 0.6192, 1.0000],
#     经过对比：节点重要性计算的g2边的权重基本都有变化--------+TEXTSIM=0.6====》进一步筛选>=TEXTSIM阈值的短文本-短文本边==》以更新g1图得到g2图
    wei=weights.tolist()
    count=0
    greatinds=[]
    greatvalues=[]
    for i in range(len(wei)):
#         if wei[i]<textsim:
        if wei[i]>=textsim:
#             print() #0.6213343739509583, 0.6304726600646973, 0.641057550907135, 0.507516622543335, 0.534453809261322,...
#         else: #满足条件的下标和对应的值存储下来
            count=count+1
            greatinds.append(i)
            greatvalues.append(wei[i])
    #取出符合条件的edge_index重新组合》符合阈值条件的新边下标+结合对应值=》组合成networkx=》转换成dgl形式的train_gg图（对应g2图）
# 442000 494580 [0.3147852122783661,---testeNeigh
# tensor(4.7005) tensor(2.7953) tensor(0.)
# 4451 6408
# 6408 6408 --len(edge_index[0]),len(edge_index[1])  print(len(edges_weights)) 5294--g2
# test:442000 494580
# 494580 494580
# 449014
# # 933 1312 --textsim1/2/3：设置为0.7，过滤g1中更多的边，g2保留少一些g1的边
# 1312 1312
# 933 933
# 973 973
# 947
# val graph： 2886 6408
# 6408 6408
# 2886 2886
# 3886 3886
# 3823
# test：339006 494580  train g2,val g2,test g2都在初始时融合一次--“可根据各自textsim阈值调整g2边数”
# g1，g2随着DFGAT模型学习，边的注意力系数随时改变---》g3图中边每个epoch发生变化（动态的g3图）
# 494580 494580
# 339006 339006
# 348628 348628
# 347515
#     print(Stop)
    edge_index= edge_index.tolist() # edge_index_:无向图
    print(len(edge_index[0]),len(edge_index[1])) #1312 1312  2592 2592 注意括号
    new_rows=[]
    new_cols=[]
    for j in range(len(greatinds)): #
        new_rows.append(edge_index[0][greatinds[j]]) #greatinds[5]=6
        new_cols.append(edge_index[1][greatinds[j]])
    print(len(new_cols),len(new_rows)) #1295 1295
#g = dgl.add_self_loop(g)----添加自循环:0 0, 1 1, 2 2,...,若没有出现0-0,1-1等类似的一对，则添加该对和对应的权重值1.0
    for j in range(numnodes): #0-6919, 加上所有对角线的元素，如0-0,1-1,2-2：确保所有节点都在新的dgl图g2中
#         for k in range(len(new_rows)):
#             if new_rows[k]!=j and new_cols[k]!=j: #0-0，相同下标的值已在行标new_rows list和new_cols list中同时存在
        new_rows.append(j) #greatinds[5]=6 ----！！！注意：对角线的weight-g2图上仍然可以保留为1.0
        new_cols.append(j)
        greatvalues.append(1.0)
    print(len(new_cols),len(new_rows))  #1335 1335
#     print(Stop)
    #組合成networkx
# 经过观察：observation， 0-0C边的权重值为0.76，“而不是1.0”-------由于节点重要性估计的边权重和相似度算法估计的权重原理不同
    nxcontents=[]
    edges_weights=dict()
    for i in range(len(new_rows)):
        templs=[]
        templs.append(new_rows[i]) #对应行
        templs.append(new_cols[i]) #对应列
        temptuple=tuple(templs) 
#         print(temptuple) #(0, 0)
        nxcontents.append(temptuple)
        edges_weights[str(temptuple)]=greatvalues[i]
#y元组形式的list--》创建DiGraph（networkx版的）
    g2_graph=nx.DiGraph(nxcontents) #创建networkx图
    g2=dgl.from_networkx(g2_graph)
#     print(len(edges_weights),edges_weights) #1296 {'(0, 0)': 1.0, '(0, 1)': 0.40300825238227844, '(0, 2)': 
#     print(Stop)
    print(len(edges_weights)) #1296 (1335中有部分类似0-0边重复）
#     print(Stop)
    return edges_weights, g2  #返回构造好的g2图

if __name__ == "__main__":
#     print(f"Training PPI Dataset with DGL built-in GATConv module.") #dgl内置的GAT模块
# data:  Data(x=[18333, 6805], edge_index=[2, 163788], y=[18333])
# data.edge_index:  tensor([[    0,     0,     0,  ..., 18331, 18331, 18332],
#         [ 5111, 12716, 12963,  ..., 14816, 17748,  2582]])
    print(f"Training My Dataset with DGL built-in GATConv module.")
#     parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset='tagmynews'
    # lujing="agnews-mymethod"
    dataset='sst2'
#     sys.stdout = Logger(logdir + "{}-TrainValTestFusionOnceRemoveMLPmodule_pretest1230.log".format(dataset))
    sys.stdout = Logger(logdir + "{}-MSGCL-D2FSF_Onlyg2Ablation.log".format(dataset)) #1230应该是融合所有的
    path2 = './data/'
    path = path2+'{}'.format(dataset)  # label = g.ndata['label']
#     sys.stdout = Logger(logdir + "{}-FusionOnceeNeighg1∩g2-g3Dynamictrainothersim-totallossxiuzheng-addCrossAtt+GCLmodule_OnlyTrainBP-TrainSupervisedGCL+UpdatePara.log".format(dataset))
#     cate=2

    numnodes=8741 #当前节点总数
    out_size = 2#分类的类别数
    current_labels=[0,1]
    tau=0.4 #GSL loss中的tau值设置 GSL loss1-->(-logxx)
    lamb=0.8 #损失系数，L=L1+lamb*L2 (L1为分类loss，L2为GSL loss）
#     textsim1=0.1 #train图，g2的边相似度
    traintextsim=0.1
    othertextsim=0.1
    learningrate=0.003
    traing2_textsim=0.6  #必须设置为固定值0.6--保持g1，g2图的差异
    # trainp=[0.6]
    # otherp=[0.55]
    # WDs=[0.06]
    train_p=0.35
    other_p=0.15
    WD=0.35
# Update_edgeindexg3, edge_attrg3=dropout_adj(edgeIndex_g3, p=0.2) #复用torcch_geometric.utils--drop_adj代码，train:0.1--414*0.1=41.4----414-41.4=373
#   p:train_p,other_p(train,val/test设置两组随机移除边概率)
#     textsim1=0.7
#     textsim2=0.6
#     textsim2=0.7 #g2的都一律用textsim2
    trainnodes=10*out_size
    valnodes=1000
#     textsim3=0.65 #test图，g2的边相似度
#     textsim3=0.6 #g3图的train,val,test都用textsim3

#     初始化GCN编码器和图对比学习MYGCL模型,传入初始化参数，，。。。
#     parser.add_argument('--device', type=str, default='cpu')#GPU1
#     default_param = { #14x项    --*.json中10项
#         'learning_rate': 0.01,
    featurein_channels=384 #   featurein_channels=384,  :注意每行语句后不能加","-->否则会误认为是tuple， 输入特征维度
    num_hidden=384
    num_proj_hidden=256 #GCL-输出特征维度和特征--》输入到DFGAT模型分类
    activation='prelu'
    base_model='GCNConv' #配置的编码器-GCN模型---在*.json文件没有
    num_layers=2 #GCN的层数--在 *.json文件没有
#         'drop_edge_rate_1': 0.3,
#         'drop_edge_rate_2': 0.4,
#         'drop_feature_rate_1': 0.1,
#         'drop_feature_rate_2': 0.0,
    tau=0.4 #是什么参数？
    k=2
    # learning_rate= 0.01
    GCL_weight_decay=1e-5
#         'num_epochs': 3000, #coauthor_cs.json配置文件中默认 "num_epochs": 1000
#         'weight_decay': 1e-5, #最后两个参数在*.json文件中没有
#         'drop_scheme': 'degree', #配置删边的参数--从pr，degree，evc三种方案中选择一种实现图上的边增强
#     }
     # add hyper-parameters into parser
#     param_keys = default_param.keys()
#     print("param_keys:",param_keys)#param_keys: dict_keys(['learning_rate', 'num_hidden', 'num_proj_hidden', 'activation', 'base_model', 'num_layers', 'drop_edge_rate_1', 'drop_edge_rate_2', 'drop_feature_rate_1', 'drop_feature_rate_2', 'tau', 'num_epochs', 'weight_decay', 'drop_scheme'])
#     for key in param_keys:
#         print("key:",key)#参数列表，key: featurein_channels
#         parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
#     args = parser.parse_args() #将配置好的参数-添加到解析器中
#     param = spparam(source=args.param, preprocess='nni') #nni包预处理npz数据集
#     print("param['featurein_channels']: ",param['featurein_channels'])
#     print("get_activation(activation):",get_activation(activation))
    print(" GCNConv: ", GCNConv) # GCNConv:  <class 'torch_geometric.nn.conv.gcn_conv.GCNConv'>
    print("feature_inchannels: ",featurein_channels)
    encoder = Encoder(featurein_channels, num_hidden, nn.PReLU(),
                      base_model= GCNConv, k=2).to(device) #k为numer_layers
    #GRACE模块，包括：编码器、投影器、对比损失，正则化等------图对比学习模块
    print("encoder: ",encoder)
    print(get_activation(activation), GCNConv)
#     print("param['num_hidden']: ",param['num_hidden'])
#     print("param['num_proj_hidden']: ", param['num_proj_hidden'])#投影隐层的维度，param['num_proj_hidden']:  256
#     print("param['tau']: ",param['tau'])#param['tau']:  0.4
    #GRACE模型将GCN编码器作为参数传入，。。。。+自身的MLP参数（输入层，隐层和输出层）+lambda指数函数的参数tau
    GCL_model = MYGCL(encoder, num_hidden, num_proj_hidden, tau).to(device)
    # GCL_optimizer = torch.optim.Adam( #配置模型的优化器参数，包括学习率、权重衰减。----届时可以调参
    #     GCL_model.parameters(),
    #     lr=learning_rate,
    #     weight_decay=weight_decay
    # )
    #1.d加载数据集 load and preprocess datasets 
    #加载label
# ) 1/j加载训练集
    with open(path+'/traindata/'+'{}_graph_network.pkl'.format(dataset), 'rb') as f:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        g1=pickle.load(f)
    train_g1=dgl.from_networkx(g1) #batched_graph.ndata['feat'],batched_graph.ndata['label']:
    print("train_g1:",train_g1)
#构造图g2：dgl版的图--》pyg版的图----取出边下标（包括行标和列标）或者直接在构图时保存边下标和对应的值
#     train_gg = from_dgl(train_g)   #g = to_dgl(data) :pyg->dgl图;ImportError: cannot import name 'from_dgl' from 'torch_geometric.utils' (J:\Anaconda3\lib\site-packages\torch_geometric\utils\__init__.py)
    #g1图中的边下标和对应的每条边对应的weight值
    with open(path+'/traindata/'+'{}_TrainEdgeIndex.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        edge_index=pickle.load(ff)
    with open(path+'/traindata/'+'{}_TrainEdgeIndex_Values.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        edge_index_values=pickle.load(ff)
    print(edge_index,edge_index_values) #g1图中的边和对应的值
#
    print(edge_index[1][2]) #1,2: tensor(2)
#     print(Stop)
   #根据边下标edge_index计算节点的中心性、边的中心性、边的权重等
  
    
    traing2_edgesweight,train_g2=Construct_Graph2Edges(edge_index, traing2_textsim,trainnodes)
   
    print("train_g2:",train_g2)
#     train_g2: Graph(num_nodes=80, num_edges=1971,  
#       ndata_schemes={}  2017-1971=46,46个已经出现了,类似0-0,1-1这样的模式？
#       edata_schemes={})+self-connection
#     print(Stop)
    #加载dataset的features
    with open(path+'/traindata/'+'{}_trainfeatures.pkl'.format(dataset), 'rb') as f2:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        feature=pickle.load(f2)
#     feature=feature.float()
    print(feature,feature.shape) #   dtype=torch.float64) torch.Size([80, 384])
    train_g1.ndata["feat"]=feature #batched_graph:第1个图对象
    train_g2.ndata["feat"]=feature
    print("train_g1:",train_g1)
    print("train_g2:",train_g2)
#
    #加载dataset的labels
#     with open(path+'/KNN/'+'{}_labels.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
    with open(path+'/traindata/'+'{}_trainlabels.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        lab=pickle.load(ff)
    print("lab: ",lab) 
    print("lab shape: ",lab.shape)
# lab:  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  lab shape:  torch.Size([40])
    #开始统计每个类别的下标list
    alltraincates=[]
    labls=lab.tolist()
    for c in range(out_size):
       currentcates=[]
       for cc in range(len(labls)):
           if labls[cc]==c:
               currentcates.append(cc)
       alltraincates.append(currentcates)
       print(len(currentcates)) #20 20
    print(alltraincates) #[[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37, 38, 39], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 26, 27, 28, 29, 30, 31, 32]]
#     print(stop)
#     lab:  tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
#         2, 2, 2, 3, 3, 3, 3, 3])
# lab shape:  torch.Size([80])
    train_g1.ndata["label"]=lab #batched_graph:第1个图对象 ,train graph对象
    print("train_g1:",train_g1)
    train_g2.ndata["label"]=lab #batched_graph:第1个图对象 ,train graph对象
    print("train_g2:",train_g2)
#！！+自连接边的新的train图g2： train_g2: Graph(num_nodes=80, num_edges=1971,
#       ndata_schemes={'feat': Scheme(shape=(300,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={})----1937+80=2017(有34个新的，2017-1971=46个本来就有)
#     print(stop)
   
  

    GCL_optimizer = torch.optim.Adam( #配置模型的优化器参数，包括学习率、权重衰减。----届时可以调参
        GCL_model.parameters(),
        lr=learningrate, #第一次调参确定的
                # lr=learning_rate,
        weight_decay=GCL_weight_decay
    )
                #2/j加载val数据集
    with open(path+'/valdata/'+'{}_graph_network.pkl'.format(dataset), 'rb') as ff2:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        valg1=pickle.load(ff2)
    val_g1=dgl.from_networkx(valg1) #batched_graph.ndata['feat'],batched_graph.ndata['label']:
    print("val_g1:",val_g1)

#g1图中的边下标和对应的每条边对应的weight值
    with open(path+'/valdata/'+'{}_ValEdgeIndex.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        edge_index2=pickle.load(ff)
    with open(path+'/valdata/'+'{}_ValEdgeIndex_Values.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        edge_index_values2=pickle.load(ff)
    print(edge_index2,edge_index_values2) #g1图中的边和对应的值

    valg2_edgesweight,val_g2=Construct_Graph2Edges(edge_index2,othertextsim,valnodes)
    print("val_g2:", val_g2)
            # Graph(num_nodes=1000, num_edges=327503,
            #       ndata_schemes={}
            #       edata_schemes={}) ！！！+自连接的新的val图g2：
            #     print(Stop)

    #加载dataset的features
    with open(path+'/valdata/'+'{}_valfeatures.pkl'.format(dataset), 'rb') as fff2:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        feature2=pickle.load(fff2)
            #     feature=feature.float()
    print(feature2,feature2.shape) #  
            #     tensor([[-0.1400, -0.1659,  0.2458,  ..., -0.1732, -0.1161,  0.3121],
            #         [-0.0834, -0.1958,  0.1903,  ..., -0.0137,  0.0411,  0.1264],
            #         [ 0.1662, -0.1360, -0.0543,  ..., -0.2146,  0.1257, -0.0857],
            #         ...,
            #        dtype=torch.float64) torch.Size([1000, 384])
            #     print(Stop)
    val_g1.ndata["feat"]=feature2 #batched_graph:第1个图对象
    val_g2.ndata["feat"]=feature2  #与g1图-验证图共享特征
    print("val_g1:",val_g1)
    print("val_g2:",val_g2)
            #     with open(path+'/KNN/'+'{}_labels.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
    with open(path+'/valdata/'+'{}_vallabels.pkl'.format(dataset), 'rb') as fff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        lab2=pickle.load(fff)
    print("lab2: ",lab2) 
    valonehotlabs=encode_onehot(lab2)
    print(valonehotlabs,valonehotlabs.shape)
            #     lab2:  tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    print("lab2 shape: ",lab2.shape) #lab2 shape:  torch.Size([1000])
            #     [[0 0 0 ... 0 1 0]
            #  [0 0 0 ... 0 1 0]
            #  [0 0 0 ... 0 1 0]
            #  ...
            #  [0 0 0 ... 1 0 0]
            #  [0 0 0 ... 1 0 0]
            #  [0 0 0 ... 1 0 0]] (1000, 8)
            #     print(stop)
    val_g1.ndata["label"]=lab2 #batched_graph:第1个图对象 ,train graph对象
    val_g2.ndata["label"]=lab2
    print("val_g1: ",val_g1)
    print("val_g2: ",val_g2)
    #3/j加载test数据集
    with open(path+'/testdata/'+'{}_graph_network.pkl'.format(dataset), 'rb') as fff3:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        testg1=pickle.load(fff3)
    test_g1=dgl.from_networkx(testg1) #batched_graph.ndata['feat'],batched_graph.ndata['label']:
    print("test_g1:",test_g1)
    #加载dataset的features
    with open(path+'/testdata/'+'{}_TestEdgeIndex.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        edge_index3=pickle.load(ff)
    with open(path+'/testdata/'+'{}_TestEdgeIndex_Values.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        edge_index_values3=pickle.load(ff)
    print(edge_index3,edge_index_values3) #g1图中的边和对应的值
    print(edge_index_values3.shape)  #torch.Size([3303252])

    testnodes=numnodes-feature.shape[0]-feature2.shape[0]
    testg2_edgesweight,test_g2=Construct_Graph2Edges(edge_index3,othertextsim,testnodes)
    print("test_g2:", test_g2)
    with open(path+'/testdata/'+'{}_testfeatures.pkl'.format(dataset), 'rb') as ffff3:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        feature3=pickle.load(ffff3)
            #     feature=feature.float()
    print(feature3,feature3.shape) #  
            #     tensor([[-0.1621,  0.6315,  0.3570,  ..., -0.5421, -0.0835,  0.1179],
            #        dtype=torch.float64) torch.Size([5860, 384])
    test_g1.ndata["feat"]=feature3 #batched_graph:第1个图对象
    test_g2.ndata["feat"]=feature3 #    " Got %d and %d instead." % (nfeats, num_nodes)
            # dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got 6920 and 6916 instead.
            #   val_g2.ndata["feat"]=feature2  
    print("test_g1:",test_g1)
    print("test_g2:",test_g2)
                #加载dataset的labels
    with open(path+'/testdata/'+'{}_testlabels.pkl'.format(dataset), 'rb') as ffff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        lab3=pickle.load(ffff)
    print("lab3: ",lab3) 
    test_g1.ndata["label"]=lab3 #lab3:  tensor([1, 2, 1,  ..., 1, 3, 2])
    test_g2.ndata["label"]=lab3 #
    testonehotlabs=encode_onehot(lab3)
    print(testonehotlabs,testonehotlabs.shape)
            # lab3:  tensor([0, 0, 0,  ..., 3, 3, 3])
            # lab3 shape:  torch.Size([6960])
    print("lab3 shape: ",lab3.shape)
    print(test_g1,test_g2)
    in_size = feature.shape[1] #384

            #     out_size = lab.shape[1]
    print("in_size: ",in_size) #in_size:  300
    print("out_size: ",out_size) #out_size:  4

            #     print(Stop)
                #3，定义模型，并实例化---自注意力-》规范化的注意力系数--》计算multi-head注意力系数，在"中间层拼接计算的多头注意力”得到节点表示----》最后一层/预测层取“均值“更新节点表示
            #model1对象用于图融合，model2对象用于模型的训练和测试。。。。
            #注意，修改下GAT模型的名称：本文提出的新方法--双级融合的图注意力模型（DFGAT)
            #     model = DFGAT(in_size, 256, out_size, heads=[4, 4, 6],GFusion=True).to(device) #默认不融合
            #g3的特征维度和g1，g2相同---》和g1.g2共享同一个DFGAT模型，同时更新模型参数
    model =DGF_MSGCL(in_size, 96, out_size, heads=[4, 4, 6]).to(device) #96*4=384
            #     model = DFGAT(in_size, 256, out_size, heads=[4, 4, 6]).to(device) #默认不融合--默认为整个DFGAT模型，在forward函数中确定是否融合
            #     model1 = DFGAT(in_size, 256, out_size, heads=[4, 4, 6],GFusion=True).to(device) #GCN每层的multi-head数分别为4,4,6
                #model2--模型输入特征维度为train_g3的特征维度

    print("model : ",model)
            #   当前自己数据集构造的模型：  model :  GAT(
            #
                # model training
    print("GraphFusion and Training...") #2， 加载训练数据
            #     train_dataloader = GraphDataLoader(train_dataset, batch_size=2) #train图20个，每批2个图--20/2=10批
            #     val_dataloader = GraphDataLoader(val_dataset, batch_size=2)
    #只用1个图训练，1个图测试
    train_dataset_list=[]
    val_dataset_list=[]
    # train_dataset_list.append(train_g1)
    # val_dataset_list.append(val_g1)
    #添加g2图
    train_dataset_list.append(train_g2) #换成g2图单独测试
    val_dataset_list.append(val_g2)
    print("train_dataset_list:",train_dataset_list)
            #     print(stop)
    #4.t图融合阶段：节点融合和边融合--只需要输入层和隐层两层-----此时已经实例化，+“真实的g1-train,val,test"
  
    test_dataset_list=[]
    # test_dataset_list.append(test_g1)
    #添加g2图
    test_dataset_list.append(test_g2) #g2图单独test
            #     test_dataset_list.append(test_dataset[0])
            #     test_dataloader = DataLoader(data, batch_size=1)
            #     test_dataloader = GraphDataLoader(data, batch_size=1)
    train_dataloader = GraphDataLoader(train_dataset_list, batch_size=1) #train图20个，每批2个图--20/2=10批
    val_dataloader = GraphDataLoader(val_dataset_list, batch_size=1)
    test_dataloader = GraphDataLoader(test_dataset_list, batch_size=1) #2个测试图
    print("train_dataloader: ",train_dataloader) #train_dataloader:  <dgl.dataloading.dataloader.GraphDataLoader object at 0x000002C3DD976F28>
#     print(stop)
    #4，训练模型  
#     train(train_dataloader, val_dataloader, device, model) #RuntimeError: mat1 and mat2 shapes cannot be multiplied (8040x384 and 50x1024)
#     print("train_dataloader:",train_dataloader) #train_dataloader: <dgl.dataloading.dataloader.GraphDataLoader object at 0x0000025DFF8E0BE0>
#     print("val_dataloader： ",val_dataloader) #val_dataloader：  <dgl.dataloading.dataloader.GraphDataLoader object at 0x0000025DD094A7F0>
#     print(STOP)
    #5，测试模型 test the model
    print("Testing...")
#     test_dataloader = GraphDataLoader(test_dataset, batch_size=2) #2个测试图
   
    train(train_dataloader, val_dataloader, device, model)
    endtime=time.time() 
    print("cost time:",endtime-starttime)
#     