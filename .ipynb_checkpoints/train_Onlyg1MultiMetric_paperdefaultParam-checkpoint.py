import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader # torch.utils.data.dataloader
# from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,accuracy_score
# from utils2 import load_data
from torch_geometric.data import Data #pgl版本的
import os, gc, sys
from print_log import Logger
import pickle
import networkx as nx
import dgl
import os, gc, sys
from print_log import Logger
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import accuracy_score
from copy import deepcopy
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
# dataset='tagmynews'
# sys.stdout = Logger(logdir + "{}.log".format(dataset))  #写入日志文件中  

class GAT(nn.Module): #模型中构造函数初始化---GraphSAGE
    def __init__(self, in_size, hid_size, out_size, heads): #attention中的multi-head注意力
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu)
        )
        self.gat_layers.append( #2层GAT层
            dglnn.GATConv( #GATConv中的方法--gatconv.py中的dgl.nn.pytorch.conv库中的函数。
                hid_size * heads[0],
                hid_size,
                heads[1],
                residual=True,
                activation=F.elu,
            )
        )
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

if __name__ == "__main__":
#     print(f"Training PPI Dataset with DGL built-in GATConv module.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset='tagmynews'
    lujing='tagmynews-SIF'
#     current learning rate:0.001
# current Weight decay:0.003
#     dataset='twitter'
    out_size =7#分类的类别数
    path2 = './data/'
    path = path2+'{}'.format(lujing)  # label = g.ndata['label']
    sys.stdout = Logger(logdir + "{}-GAT-SIFDuibi2-2.log".format(dataset))
    #1.加载数据集 load and preprocess datasets 
#     train_dataset = PPIDataset(mode="train") #创建Dataset函数，包含若干个组件，
#     val_dataset = PPIDataset(mode="valid")
#     test_dataset = PPIDataset(mode="test")
#     features = train_dataset[0].ndata["feat"]
#     print("train_dataset:",train_dataset) #train_dataset: Dataset("ppi", num_graphs=20, save_path=C:\Users\Administrator\.dgl\ppi_4b14ad03)
#     print("val_dataset: ",val_dataset) #val_dataset:  Dataset("ppi", num_graphs=2, save_path=C:\Users\Administrator\.dgl\ppi_4b14ad03)
#     print("test_dataset: ",test_dataset) #test_dataset:  Dataset("ppi", num_graphs=2, save_path=C:\Users\Administrator\.dgl\ppi_4b14ad03)
#     print("features: ",features)
#     print("features shape: ",features.shape) #features shape:  torch.Size([1767, 50])
#     features:  tensor([[-0.0855, -0.0884, -0.1128,  ..., -0.1399, -0.1494, -0.1481],
#         [-0.0855, -0.0884, -0.1128,  ..., -0.1399, -0.1494, -0.1481],
#         [-0.0855, -0.0884, -0.1128,  ..., -0.1399, -0.1494, -0.1481],
#         ...,
#         [-0.0855, -0.0884, -0.1128,  ..., -0.1399, -0.1494, -0.1481],...
#     print("train_dataset[0]: ",train_dataset[0]) #第0个图：节点数，边数，邻接矩阵，特征，标签
#     train_dataset[0]:  Graph(num_nodes=1767, num_edges=34085,
#       ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64), 'feat': Scheme(shape=(50,), dtype=torch.float32), 'label': Scheme(shape=(121,), dtype=torch.float32)}
#       edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})--对应某个图
#     print(Stop)
    # create GAT model
    #将dataset中的特征维数、标签数、类别数都换成自己的
#     adj, features_1,edge_indices_list,adj_values_list,features_indices_list = load_data(path=path+'/KNN/', dataset=dataset)
#     print("adj: ",adj)
#     print("features_1 " ,features_1)
#     Num of edges: 348226
# adj:  [[tensor(indices=tensor([[   0,  110,  116,  ..., 7907, 7962, 8039],
#                        [   0,    0,    0,  ..., 8039, 8039, 8039]]),
#        values=tensor([0.0405, 0.0158, 0.0179,  ..., 0.0223, 0.0187, 0.0732]),
#        size=(8040, 8040), nnz=348226, layout=torch.sparse_coo)]]
# features_1  [tensor(indices=tensor([[   0,    0,    0,  ..., 8039, 8039, 8039],
#                        [   0,    1,    2,  ...,  381,  382,  383]]),
#        values=tensor([ 0.0661,  1.1931,  0.1492,  ...,  0.2038,  0.1264,
#                       -0.0145]),
#        size=(8040, 384), nnz=3087360, layout=torch.sparse_coo)]
#     print(stop)
#     textfeatures=features_1[0].to_dense()
#     print("textfeatures: ",textfeatures)
#     print("textfeatures shape: ",textfeatures.shape) #textfeatures shape:  torch.Size([8040, 384])
#     textfeatures:  tensor([[ 0.0661,  1.1931,  0.1492,  ...,  0.1259, -0.2615, -1.2531],
#         [-0.1433,  0.0363,  0.0367,  ..., -0.0801, -0.0977, -0.1698],
#         [-0.0129,  0.2250,  0.0203,  ...,  0.2687,  0.0277, -0.1118],
#         ...,
#         [ 0.0018, -0.1438,  0.0112,  ..., -0.0726, -0.0210,  0.1347],
#         [ 0.0439,  0.2105, -0.3676,  ...,  0.0255,  0.1767, -0.3309],
#         [ 0.0354, -0.0233, -0.1578,  ...,  0.2038,  0.1264, -0.0145]])
    #加载label
#     with open(path+'/KNN/'+'/{}_data.y.pkl'.format(dataset),'rb') as file:  
#         data_y=pickle.load(file) 
#     print("data_y: ",data_y) #data_y:  [0 0 1 ... 1 1 1]
#     print("data_y shape: ",data_y.shape) #data_y shape:  (8761,)
#     data_y=torch.from_numpy(data_y).type(torch.LongTensor)
#     print(data_y) #tensor([0, 0, 0,  ..., 3, 3, 3])
# #     print(stop)
#     data = Data(x=textfeatures,y=data_y, edge_index=edge_indices_list[0]) #没有传入节点的label，labels = batched_graph.ndata["label"].float()
#     in_size = features.shape[1]
#     out_size = train_dataset.num_labels
#     #3，定义模型，并实例化
#     model = GAT(in_size, 256, out_size, heads=[4, 4, 6]).to(device)
#     print("in_size: ",in_size) #in_size:  50
#     print("out_size: ",out_size) #out_size:  121
#     print("model : ",model)
#     model : model :  GAT(
#   (gat_layers): ModuleList(
#     (0): GATConv(    
#       (fc): Linear(in_features=50, out_features=1024, bias=False)
#       (feat_drop): Dropout(p=0.0, inplace=False)
#       (attn_drop): Dropout(p=0.0, inplace=False)
#       (leaky_relu): LeakyReLU(negative_slope=0.2)
#     )
#     (1): GATConv(
#       (fc): Linear(in_features=1024, out_features=1024, bias=False)
#       (feat_drop): Dropout(p=0.0, inplace=False)
#       (attn_drop): Dropout(p=0.0, inplace=False)
#       (leaky_relu): LeakyReLU(negative_slope=0.2)
#       (res_fc): Identity()
#     )
#     (2): GATConv(
#       (fc): Linear(in_features=1024, out_features=726, bias=False)
#       (feat_drop): Dropout(p=0.0, inplace=False)
#       (attn_drop): Dropout(p=0.0, inplace=False)
#       (leaky_relu): LeakyReLU(negative_slope=0.2)
#       (res_fc): Linear(in_features=1024, out_features=726, bias=True)
#     ) 121*6=726
#   )
# ) 1/j加载训练集   
    with open(path+'/traindata/'+'{}_graph_network.pkl'.format(dataset), 'rb') as f:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        g1=pickle.load(f)
    train_g=dgl.from_networkx(g1) #batched_graph.ndata['feat'],batched_graph.ndata['label']:

#     Graph(num_nodes=8040, num_edges=348226,
#       ndata_schemes={}
#       edata_schemes={})
   #加载dataset的features
    with open(path+'/traindata/'+'{}_trainfeatures.pkl'.format(dataset), 'rb') as f2:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        feature=pickle.load(f2)
        print("train_g:",train_g)
    print(train_g.edges())
    train_edgeindexgrow=train_g.edges()[0].tolist()#行
    train_edgeindexgcol=train_g.edges()[1].tolist()#列
    print(train_g.nodes(),train_edgeindexgrow,train_edgeindexgcol)
#     tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
#         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
#     print(Stop)
    train_UpdatedGNodes=train_g.nodes().tolist()
    print(train_UpdatedGNodes) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    #     print(Crossattg3[0],Crossattg3[0].shape) 
    AddallNodes_trainEdgeIndex=[]
    AddallNodes_trainEdgeIndex.append(train_edgeindexgrow)
    AddallNodes_trainEdgeIndex.append(train_edgeindexgcol)
#     print(z1.shape[0]) #40
#     print(Stop)
    for i in range(feature.shape[0]): #在当前fea所有节点查找当前更新的图元素
    #         print(torch.tensor(i)) #tensor(39)
    #         print(i)
        if i not in train_UpdatedGNodes: 
            print("no exist..")
            print(i)
            train_edgeindexgrow.append(i) 
            train_edgeindexgcol.append(i)
            AddallNodes_trainEdgeIndex[0].append(i)
            AddallNodes_trainEdgeIndex[1].append(i)
    train_g = dgl.graph((torch.tensor( train_edgeindexgrow), torch.tensor(train_edgeindexgcol))) #加入缺少的节点后重新构图
#     feature=feature.float()
# train_g: Graph(num_nodes=70, num_edges=159,
#       ndata_schemes={'feat': Scheme(shape=(904,), dtype=torch.float64)}
#       edata_schemes={})
    print(feature,feature.shape) #   dtype=torch.float64) torch.Size([80, 384])
#     tensor([[-0.0040, -0.1768,  0.2434,  ..., -0.1804, -0.3630,  0.2488],
#         [ 0.0735, -0.1637, -0.0454,  ..., -0.1746, -0.1356,  0.1198],
#         [-0.0877, -0.0848, -0.0250,  ..., -0.1961,  0.2767, -0.1000],
#         ...,
#         [-0.0041,  0.3287, -0.0255,  ...,  0.1661,  0.0480, -0.3080],
#         [-0.0416, -0.1997,  0.3488,  ..., -0.0242, -0.1677,  0.3140],
#         [-0.0581,  0.0383,  0.2592,  ..., -0.3347, -0.2076,  0.0238]],
#        dtype=torch.float64) torch.Size([80, 384])
#     print(Stop) dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got 70 and 67 instead.

    train_g.ndata["feat"]=feature #batched_graph:第1个图对象
    print("train_g:",train_g)
#     train_g: Graph(num_nodes=8040, num_edges=348226,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float32)}
#       edata_schemes={})
# train_g: Graph(num_nodes=20, num_edges=37,
#       ndata_schemes={'feat': Scheme(shape=(1535,), dtype=torch.float64)}
#       edata_schemes={})
# '#     print(stop)'
    #加载dataset的labels
#     with open(path+'/KNN/'+'{}_labels.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
    with open(path+'/traindata/'+'{}_trainlabels.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        lab=pickle.load(ff)
#     print("lab: ",lab) #lab:  [0 0 0 ... 3 3 3],lab shape:  (8040,)
#     print("lab shape: ",lab.shape) #lab shape:  torch.Size([8040, 4]) 
#     lab:  tensor([[1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         [1., 0., 0., 0.],
#         ...,
#         [0., 0., 0., 1.],
#         [0., 0., 0., 1.],
#         [0., 0., 0., 1.]])
#     lab=torch.tensor(lab)
    print("lab: ",lab) 
    print("lab shape: ",lab.shape)
#     lab:  tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
#         2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1,
#         1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2,
#         2, 2, 2, 3, 3, 3, 3, 3])
# lab shape:  torch.Size([80])
# train_g: Graph(num_nodes=80, num_edges=2750,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={})
# lab:  tensor([0, 0, 0,  ..., 3, 3, 3], dtype=torch.int32)
# lab shape:  torch.Size([8040])
# train_g: Graph(num_nodes=8040, num_edges=348226,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int32)}
#       edata_schemes={})
    train_g.ndata["label"]=lab #batched_graph:第1个图对象 ,train graph对象
    print("train_g:",train_g)
#  赋值后：   train_g: Graph(num_nodes=80, num_edges=2750,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={})
    #2/j加载val数据集
    with open(path+'/valdata/'+'{}_graph_network.pkl'.format(dataset), 'rb') as ff2:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        g2=pickle.load(ff2)
    val_g=dgl.from_networkx(g2) #batched_graph.ndata['feat'],batched_graph.ndata['label']:
    print("val_g:",val_g)
#     val_g: Graph(num_nodes=1000, num_edges=40806,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64)}
#       edata_schemes={})
#     Graph(num_nodes=8040, num_edges=348226,
#       ndata_schemes={}
#       edata_schemes={})
   #加载dataset的features
    with open(path+'/valdata/'+'{}_valfeatures.pkl'.format(dataset), 'rb') as fff2:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        feature2=pickle.load(fff2)
    print(val_g.edges())
    val_edgeindexgrow=val_g.edges()[0].tolist()#行
    val_edgeindexgcol=val_g.edges()[1].tolist()#列
    print(val_g.nodes(),val_edgeindexgrow,val_edgeindexgcol)
#     tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
#         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    val_UpdatedGNodes=val_g.nodes().tolist()
    print(val_UpdatedGNodes) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    #     print(Crossattg3[0],Crossattg3[0].shape) 
    AddallNodes_valEdgeIndex=[]
    AddallNodes_valEdgeIndex.append(val_edgeindexgrow)
    AddallNodes_valEdgeIndex.append(val_edgeindexgcol)
#     print(z1.shape[0]) #40
#     print(Stop)
    for j in range(feature2.shape[0]): #在当前fea所有节点查找当前更新的图元素
    #         print(torch.tensor(i)) #tensor(39)
    #         print(i)
        if j not in val_UpdatedGNodes: 
            print("no exist..")
            print(j)
            val_edgeindexgrow.append(j) 
            val_edgeindexgcol.append(j)
            AddallNodes_valEdgeIndex[0].append(j)
            AddallNodes_valEdgeIndex[1].append(j)
    val_g = dgl.graph((torch.tensor( val_edgeindexgrow), torch.tensor(val_edgeindexgcol))) #加入缺少的节点后重新构图
#     feature=feature.float()
    print(feature2,feature2.shape) #  torch.Size([1000, 1535])
#     tensor([[-0.1400, -0.1659,  0.2458,  ..., -0.1732, -0.1161,  0.3121],
#         [-0.0834, -0.1958,  0.1903,  ..., -0.0137,  0.0411,  0.1264],
#         [ 0.1662, -0.1360, -0.0543,  ..., -0.2146,  0.1257, -0.0857],
#         ...,
#         [ 0.1604,  0.0643,  0.1015,  ..., -0.0039, -0.2467,  0.0869],
#         [ 0.2342,  0.0178,  0.0639,  ..., -0.1343, -0.0895,  0.0185],
#         [ 0.1297, -0.0018,  0.5251,  ..., -0.2857,  0.0468, -0.0813]],
#        dtype=torch.float64) torch.Size([1000, 384])
#     print(Stop)
    val_g.ndata["feat"]=feature2 #batched_graph:第1个图对象
    print("val_g:",val_g)
#     print(Stop)
#     with open(path+'/KNN/'+'{}_labels.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
    with open(path+'/valdata/'+'{}_vallabels.pkl'.format(dataset), 'rb') as fff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        lab2=pickle.load(fff)
    print("lab2: ",lab2) 
#     lab2:  tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    print("lab2 shape: ",lab2.shape) #lab2 shape:  torch.Size([1000])
    val_g.ndata["label"]=lab2 #batched_graph:第1个图对象 ,train graph对象
    valonehotlabs=encode_onehot(lab2)
    print(valonehotlabs,valonehotlabs.shape)
    print("val_g: ",val_g)
# val_g:  Graph(num_nodes=1000, num_edges=40806,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={})
    #3/j加载test数据集
#     train_g: Graph(num_nodes=8040, num_edges=348226,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float32), 'label': Scheme(shape=(4,), dtype=torch.float32)}
#       edata_schemes={})
    with open(path+'/testdata/'+'{}_graph_network.pkl'.format(dataset), 'rb') as fff3:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        g3=pickle.load(fff3)
    test_g=dgl.from_networkx(g3) #batched_graph.ndata['feat'],batched_graph.ndata['label']:
#     print("test_g:",test_g)
# test_g: Graph(num_nodes=6960, num_edges=300960,
#       ndata_schemes={}
#       edata_schemes={}) test_g: Graph(num_nodes=5930, num_edges=7196,
#       ndata_schemes={}
#       edata_schemes={})
   #加载dataset的features
    with open(path+'/testdata/'+'{}_testfeatures.pkl'.format(dataset), 'rb') as ffff3:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        feature3=pickle.load(ffff3)
#     print(test_g.edges())
    test_edgeindexgrow=test_g.edges()[0].tolist()#行
    test_edgeindexgcol=test_g.edges()[1].tolist()#列
#     print(test_g.nodes(),test_edgeindexgrow,test_edgeindexgcol)
#     tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
#         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    test_UpdatedGNodes=test_g.nodes().tolist()
#     print(test_UpdatedGNodes) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    #     print(Crossattg3[0],Crossattg3[0].shape) 
    AddallNodes_testEdgeIndex=[]
    AddallNodes_testEdgeIndex.append(test_edgeindexgrow)
    AddallNodes_testEdgeIndex.append(test_edgeindexgcol)
#     print(z1.shape[0]) #40
#     print(Stop)
    for k in range(feature3.shape[0]): #在当前fea所有节点查找当前更新的图元素
    #         print(torch.tensor(i)) #tensor(39)
    #         print(i)
        if k not in val_UpdatedGNodes: 
            print("no exist..")
            print(k)
            test_edgeindexgrow.append(k) 
            test_edgeindexgcol.append(k)
            AddallNodes_testEdgeIndex[0].append(k)
            AddallNodes_testEdgeIndex[1].append(k)
    test_g = dgl.graph((torch.tensor( test_edgeindexgrow), torch.tensor(test_edgeindexgcol))) #加入缺少的节点后重新构图
#     feature=feature.float()
    print(feature3,feature3.shape) #  
# tensor([[-0.0284, -0.5120, -0.0640,  ..., -0.0540,  0.1122,  0.5377],
#         [ 0.1759, -0.0445, -0.0450,  ...,  0.0983,  0.1198,  0.2083],
#         [ 0.0201, -0.3518, -0.0317,  ..., -0.4203, -0.0433,  0.1748],
#         ...,
#         [-0.2492, -0.2647,  0.3429,  ..., -0.1059,  0.1717, -0.0489],
#         [ 0.0676,  0.2461,  0.1963,  ..., -0.1377,  0.0073,  0.0811],
#         [-0.1508, -0.2481,  0.0413,  ..., -0.1013, -0.1088, -0.0420]],
#        dtype=torch.float64) torch.Size([6960, 384])
#     print(Stop)
    test_g.ndata["feat"]=feature3 #batched_graph:第1个图对象
    print("test_g:",test_g)
#     tagmynews-TFIDF: test_g: Graph(num_nodes=5930, num_edges=370646,
#       ndata_schemes={'feat': Scheme(shape=(904,), dtype=torch.float64)}
#       edata_schemes={})
#     print(Stop)
# test_g: Graph(num_nodes=6960, num_edges=300960,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64)}
#       edata_schemes={}) ---gnodes: 6960 gedges: 153960 *2=307920
    #加载dataset的labels
#     with open(path+'/KNN/'+'{}_labels.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
    with open(path+'/testdata/'+'{}_testlabels.pkl'.format(dataset), 'rb') as ffff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        lab3=pickle.load(ffff)
    print("lab3: ",lab3) 
    test_g.ndata["label"]=lab3 #batched_graph:第1个图对象 ,train graph对象
    testonehotlabs=encode_onehot(lab3)
    print(testonehotlabs,testonehotlabs.shape)
# lab3:  tensor([0, 0, 0,  ..., 3, 3, 3])
# lab3 shape:  torch.Size([6960])
    print("lab3 shape: ",lab3.shape)
    print(test_g)
#     Graph(num_nodes=6960, num_edges=300960,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={})
#     print(Stop)
    in_size = feature.shape[1] #384
   
#     out_size = lab.shape[1]
    print("in_size: ",in_size) #in_size:  384
    print("out_size: ",out_size) #out_size:  4
    #3，定义模型，并实例化
    model = GAT(in_size, 256, out_size, heads=[4, 4, 6]).to(device)
#     model = GAT(in_size, 256, out_size, heads=[4, 4, 1]).to(device)
# GRAPH ATTENTION NETWORKS Petar Veli论文中超参数默认设置
#    For the "inductive learning" task, we apply a three-layer GAT model. Both of the
# first two layers consist of K = 4 attention heads computing F0 = 256 features (for a total of 1024
# features), followed by an ELU nonlinearity. The final layer is used for (multi-label) classification:
# K = 6 attention heads computing 121 features each, that are averaged and followed by a logistic
# sigmoid activation. The training sets for this task are sufficiently large and we found no need to apply
# L2 regularization or dropout—we have, however, successfully  ----weight decay:0
    print("model : ",model)
#   当前自己数据集构造的模型：  model :  GAT(
#   (gat_layers): ModuleList(
#     (0): GATConv(
#       (fc): Linear(in_features=384, out_features=1024, bias=False)
#       (feat_drop): Dropout(p=0.0, inplace=False)
#       (attn_drop): Dropout(p=0.0, inplace=False)
#       (leaky_relu): LeakyReLU(negative_slope=0.2)
#     )
#     (1): GATConv(
#       (fc): Linear(in_features=1024, out_features=1024, bias=False)
#       (feat_drop): Dropout(p=0.0, inplace=False)
#       (attn_drop): Dropout(p=0.0, inplace=False)
#       (leaky_relu): LeakyReLU(negative_slope=0.2)
#       (res_fc): Identity()
#     )
#     (2): GATConv(
#       (fc): Linear(in_features=1024, out_features=24, bias=False)
#       (feat_drop): Dropout(p=0.0, inplace=False)
#       (attn_drop): Dropout(p=0.0, inplace=False)
#       (leaky_relu): LeakyReLU(negative_slope=0.2)
#       (res_fc): Linear(in_features=1024, out_features=24, bias=True)
#     )---out_features:4*6=24
#   )
# )
#     print(Stop)
    # model training
    print("Training...") #2， 加载训练数据
#     train_dataloader = GraphDataLoader(train_dataset, batch_size=2) #train图20个，每批2个图--20/2=10批
#     val_dataloader = GraphDataLoader(val_dataset, batch_size=2)
    #只用1个图训练，1个图测试
    train_dataset_list=[]
    val_dataset_list=[]
    train_dataset_list.append(train_g)
    val_dataset_list.append(val_g)
    #只用1个图测试
    test_dataset_list=[]
    test_dataset_list.append(test_g)
#     test_dataset_list.append(test_dataset[0])
#     test_dataloader = DataLoader(data, batch_size=1)
#     test_dataloader = GraphDataLoader(data, batch_size=1)
    
#     train_dataset_list.append(train_dataset[0])
#     val_dataset_list.append(val_dataset[0])
#     print("data: ",data)
#     data:  Data(x=[8040, 3254], edge_index=[2, 576118], y=[8040])
# train_dataloader:  <torch.utils.data.dataloader.DataLoader object at 0x000002674B730898>'
#     train_dataloader = DataLoader(data, batch_size=1)#GraphDataloader加载多个图，Dataloader可以加载一个图上的数据？
#     val_dataloader = DataLoader(data, batch_size=1)
#     train_dataloader = GraphDataLoader(data, batch_size=1)#GraphDataloader加载多个图，Dataloader可以加载一个图上的数据？
#     val_dataloader = GraphDataLoader(data, batch_size=1)
#     print("train_dataloader: ",train_dataloader) #train_dataloader:  <dgl.dataloading.dataloader.GraphDataLoader object at 0x0000014793CDC0B8>
#     print(Stop)
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
#     avg_score = evaluate_in_batches(test_dataloader, device, model) #
#     print("test_dataloader:",test_dataloader)
    print("avg_score： ",avg_score)
    print("Test Accuracy (F1-score) {:.4f}".format(avg_score))
    
    
#     data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
#                     val_mask=val_mask, test_mask=test_mask,
#                     stopping_mask=stopping_mask)

#     y = torch.tensor(data['labels'], dtype=torch.long)

#  y = dataset[0].y.view(-1).to(test_device)
# loss_fcn:  NLLLoss()
# optimizer:  Adam (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     eps: 1e-08
#     lr: 0.005
#     weight_decay: 0
# )
# Traceback (most recent call last):
#   File "I:\Inductive_SSL_ShortTextClassification\InductiveSSL_STClassification\pytorch-GAT-main--RunSuccess\pytorch-GAT-main\gat-dgl-version\train_ppi-tiaoshi.py", line 260, in <module>
#     train(train_dataloader, val_dataloader, device, model)
#   File "I:\Inductive_SSL_ShortTextClassification\InductiveSSL_STClassification\pytorch-GAT-main--RunSuccess\pytorch-GAT-main\gat-dgl-version\train_ppi-tiaoshi.py", line 127, in train
#     for batch_id, batched_graph in enumerate(train_dataloader): #KeyError: 0
#   File "J:\Anaconda3\lib\site-packages\torch\utils\data\dataloader.py", line 521, in __next__
#     data = self._next_data()
#   File "J:\Anaconda3\lib\site-packages\torch\utils\data\dataloader.py", line 561, in _next_data
#     data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
#   File "J:\Anaconda3\lib\site-packages\torch\utils\data\_utils\fetch.py", line 49, in fetch
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "J:\Anaconda3\lib\site-packages\torch\utils\data\_utils\fetch.py", line 49, in <listcomp>
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "J:\Anaconda3\lib\site-packages\torch_geometric\data\data.py", line 444, in __getitem__
#     return self._store[key]
#   File "J:\Anaconda3\lib\site-packages\torch_geometric\data\storage.py", line 85, in __getitem__
#     return self._mapping[key]
# KeyError: 0
# RuntimeError: 0D or 1D target tensor expected, multi-target not supported
# 
# pytorch 中计计算交叉熵损失函数时， 输入的正确 label 不能是 one-hot 格式。函数内部会自己处理成 one hot 格式。所以不需要输入 [ 0 0 0 0 1]，只需要输入 4 就行。
# 
# 在经过 loss 的时候，CrossEntropyLoss 会自动为其编码为 one-hot 编码，这样就会导致其升高一维。
# 
# 所以解决方案就是采用多标签问题的损失函数。比如 MultiLabelSoftMarginLoss，或者我采用的最原始的 MSELoss。