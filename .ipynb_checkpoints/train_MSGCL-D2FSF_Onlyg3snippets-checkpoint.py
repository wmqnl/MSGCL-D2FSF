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
#一阶邻域每个节点嵌入/特征重要性计算：需要传入一阶邻域所有节点的特征重要性/节点嵌入，例如，节点0,一阶邻域假设为0-0,0-1,0-2,0-10===>0节点一阶邻域集为{0,1,2,10}
# -----c通过计算0节点邻域内所有节点的特征重要性，可以得到0,1,2,10四个节点的特征重要性得分，。。接着，计算1节点一阶邻域，2节点一阶邻域，。。。直至所有节点的一阶邻域统计完成，且计算完毕
# 最终，将计算出的各个节点，如0节点，1节点，....json格式依次汇总+“加权平均”===》得到每个节点最终的特征重要性得分（此时为g1图中train图的各个节点特征重要性得分），同理，得到g2train图各个节点特征重要性得分，...
# ------》一、g1train图各节点特征重要性得分+g2train图各阶段特征重要性得分“对每个节点嵌入加权求和”【类似于self-attention求法】==》g3train图所有节点嵌入
# 二、对于g1train图和g2train图中每条边+“根据每个图得到的边级注意力得分对每个图上的边权重加权求和==》得到g3图上所有边的初始权重+“权重阈值过滤”==》得到g3train图所有边
# 三、最终，汇总得到的g3图上边-边的邻接矩阵+图上所有节点嵌入===》g3train图
# ！！！注意：根据效果可以将计算的g1图特征重要性score1和g2图特征重要性得分===》两个图汇总的重要性得分进一步规范化[0,1]之间（使用max-min规范化或其它规范化方式）
#    同理，可以汇总g1图和g2图两个图上的所有edge注意力得分==》进一步规范化[0,1]之间（使用max-min规范化或其它规范化方式）
#     def FeatureImportance_Compute(self, node_embeddings):  #C-CS注意力只需传入概念参数c即可，  2g个公式-----通过线性变换的方式将高维的向量=》映射到低维的空间,如1维--表示C-CS注意力得分。
#         node_embeddings=node_embeddings.unsqueeze(0) #在c的第一维度增加一维，使二维-》三维
#         node_embeddings= self.w(F.tanh(self.W(node_embeddings))) #将32维的概念--》1维的概念（实际为1维的注意力重要性得分）  经过两个全连接层输出概念c batch_size, concept_seq_len, 1 ，F.tanh：tanh激活函数非线性变换，self.w2，self.W2：初始化的权重参数
#         node_score = F.softmax(node_embeddings.squeeze(-1), -1) # 规范化的注意力系数，各个注意力系数总和为1  batch_size, concept_seq_len,输出beta：概念-概念集的注意力系数
#         return node_score

#     def forward(self, g, inputs): #模型前向传播函数，，不断更新节点表示
    def forward(self, g, inputs,GFusion, HiddenEmbeds): #模型前向传播函数，，不断更新节点表示
        h = inputs.float()  #正常代码，需设置gatconv.py中的 get_attention=True
#         print("--h shape: ",h.shape)
        for i, layer in enumerate(self.gat_layers): #包括输入层，隐层和输出层-----三層計算完了得到輸出曾嵌入的均值则返回最终的h值
#             h = layer(g, h) #输入图g，h参数---------》返回更新的嵌入h
            h, edge_attention = layer(g, h)  #分类阶段，GAT模型中返回的边注意力不使用
#             h = layer(g, h)
            if i == 2:  # last layer 最后一层，取mulihead注意力计算的嵌入的均值
                h = h.mean(1)
            else:  # other layer(s)--其它层，拼接不同muli-head注意力计算的嵌入
                h = h.flatten(1)
                edge_attention=edge_attention.mean(1) #对列取均值，torch.Size([2592, 4, 1])---》2592, 1
#                 if i==1 and self.GFusion==True: #保存隐层的h和edge_attention到2个pkl文件（对应边级重要性和节点级重要性）
                Crossattg3=[]
                if i==1 and GFusion==True: #图融合阶段，保存隐层的h和edge_attention到2个pkl文件（对应边级重要性和节点级重要性） 
                    if len(HiddenEmbeds)==0: #当前g1
                        HiddenEmbeds.append(h) #g1
                    elif len(HiddenEmbeds)==1: #g2,开始使用crossatten融合
#                         print(h,h.shape,HiddenEmbeds[0],HiddenEmbeds[0].shape)
#                         tensor([[ 0.3933, -0.0944, -0.0152,  ..., -0.0424, -0.3628,  0.2357],
#         [ 0.0970,  0.0413,  0.0737,  ...,  0.1235,  0.4419, -0.1669]],
#        grad_fn=<ReshapeAliasBackward0>) torch.Size([40, 1024]) tensor([[ 0.0072, -0.0320, -0.0556,  ...,  0.0417, -0.1180, -0.0153],
#         [-0.0309, -0.0566, -0.0194,  ...,  0.0518, -0.1040,  0.0005],
#        grad_fn=<ReshapeAliasBackward0>) torch.Size([40, 1024])
                        FusionAtt_features=self.crossatt(h.unsqueeze(1),HiddenEmbeds[0].unsqueeze(1)) #实例化一个对象必须和q，k，v的原始维度一样  RuntimeError: mat1 and mat2 shapes cannot be multiplied (40x1024 and 384x384)
#                         FusionAtt_features=crossatten()
#                         print(FusionAtt_features,FusionAtt_features.shape)  #torch,Size( 40,1,1024) torch.Size([1000, 1, 1024])
#                         print(Stop)
#                         Projembeddings =self.projectLayer(FusionAtt_features) #TypeError: forward() missing 1 required positional argument: 'embeddings'
#                         print(Projembeddings,Projembeddings.shape)  # torch.Size([1000, 1, 384])
#                         Projembeddings= projLayer()
#                         CrossAtt_g3features=Projembeddings.squeeze(1) 
                        CrossAtt_g3features=FusionAtt_features.squeeze(1)
                        Crossattg3.append(CrossAtt_g3features)
                    return edge_attention,h,HiddenEmbeds,Crossattg3 #隐层的g1，g2节点嵌入 AvgReadout-->g1,g2图嵌入 
        return h
    
    #GraphFusion函数写在DFGAT模型里---》DFGAT模型的一部分

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
#     strlabels=''.join()
#     print(strlabels)
#     labels_array=
#     labels_onehot=encode_onehot(newlabelsarr)
#     print(labels_onehot,labels_onehot.shape) 
    classes = set(newlabelsarr)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, newlabelsarr)),
                             dtype=np.int32)
    return labels_onehot    
#可以加上precision,recall 和F1值多个评估指标-----------++工作量
def evaluate(Up_g3, updatedg3features, dlabels,onehotlabs, model): #1个训练图，调用这个？
    GFusion=False
    model.eval()  
    correct = 0
    total = 0
#     with torch.no_grad(): # 测试时不需要梯度下降？   output = model(input_features_train, input_adj_train)#调用HGCN模型时传入所有节点的特征和adj
#     output = model(g, features, GFusion)  #调用DFGAT的forward函数 ++loss2 (val/test loss）
# TypeError: forward() missing 1 required positional argument: 'HiddenEmbeds'
    HiddenEmbeds=[] #分类阶段，该参数不使用
    output = model(Up_g3, updatedg3features, GFusion,HiddenEmbeds)  #调用DFGAT的forward函数 ++loss2 (val/test loss）
#     total += dlabels.size(0) #labels.size(0):  8040
#     print("labels.size(0): ",labels.size(0))
            ## c张量之间的比较运算
    pred2 = torch.argmax(output, 1)
#         print("pred2:",pred2) #pred2: tensor([2, 2, 2,  ..., 3, 3, 3])
#     num_correct = (pred2 == dlabels.data).sum() 
#         correct += (predicted == labels).sum().item()
#     print('accuracy: %d %% ' % (100*num_correct/total))
#  labels.data:  tensor([0, 0, 0,  ..., 3, 3, 3], dtype=torch.int32)       
#         print("pred: ",pred) #
#     Acc=100*num_correct/total
#     print("Acc:",Acc) #Acc: tensor(19.7000) score: 0.5639128214105015
#         pred = model(g, features) 
#     preds_probs = output.cpu().detach().numpy()
#     preds = deepcopy(preds_probs)
#     print(pred2,dlabels)
#     tensor([6, 6, 6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#          tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     print(pred2.shape,dlabels.shape) #torch.Size([1000]) torch.Size([1000]) TypeError: only size-1 arrays can be converted to Python scalars
# 查看classification中的相关参数类型，  y_true : 1d array-like, or label indicator array / sparse matrix
#         Ground truth (correct) target values.
# 
#     y_pred : 1d array-like, or label indicator array / sparse matrix
#         Estimated targets as returned by a classifier.

#     print(Stop)
    # [precision, recall, F1, support] = \
    #     precision_recall_fscore_support(y_true=dlabels, y_pred=pred2,labels=current_labels, average=None)
    # Accuracy= accuracy_score(dlabels, pred2) #不使用one-hot，直接使用标签数值预测
    # [precision, recall, F1, support] = \
    #     precision_recall_fscore_support(y_true=dlabels.data, y_pred=pred2.cpu().detach().numpy(),labels=current_labels, average=None)
    [precision, recall, F1, support] = \
        precision_recall_fscore_support(y_true=dlabels.data, y_pred=pred2.cpu().detach().numpy(),labels=current_labels, average='macro')
    Accuracy= accuracy_score(dlabels.data, pred2.cpu().detach().numpy()) 
    return Accuracy,precision, recall, F1,output
#     print(Accuracy,precision,recall,F1) 
#     print("mean precsion: ",precision.mean())
#     print("mean recall: ",recall.mean())
#     print("mean F1: ",F1.mean())
#   0.156 [0.71428571 0.24450549 0.5        0.         0.03753351 0.
#  0.16513761 0.33333333] [0.03968254 0.7007874  0.01086957 0.         0.4375     0.
#  0.29268293 0.0990099 ] [0.07518797 0.36252546 0.0212766  0.         0.0691358  0.
#  0.2111437  0.15267176]
# mean precsion:  0.24934945860847307
# mean recall:  0.1975665417867627
# mean F1:  0.11149265964086932----precsion，recall和F1值每类8个预测结果
#     print(' Acc2: %6.2f' % Acc2,  # Acc:  50.00 P:  25.1 R:  49.8 F1:  33.3Traceback (most recent call last):
#             'P: %5.1f' % (precision*100),
#             'R: %5.1f' % (recall*100),
#             'F1: %5.1f' % (F1*100),
#             end="")
    
#     print(stop)
    # return Accuracy,precision.mean(), recall.mean(), F1.mean(),output
#    optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#   total_loss += loss.item()

def evaluate_in_batches(g2_edgesweight,txtsim3,dataloader, device, model,edgeindex,edgeindex_values,datalabels,onehotlabs, loss_fcn,features): #评估时只传入testdataloader
#     total_score = 0 注意。评估val/test时只计算了分类acc，没有计算loss，更新参数
   #每5轮，val一次，  avg_score = evaluate_in_batches(val_dataloader, device, model2)
#    最后一轮： test_dataloader = GraphDataLoader(test_dataset_list, batch_size=1) #g3测试图
#     avg_score = evaluate_in_batches(test_dataloader, device, model2)
    #注意，GraphFusion的model和train的model分别为model1和model2；dataloader均为valdataloader或testdataloader;边下标和边值以及当前图标签可以为val/test
    totalloss=0
    alltraincates=[]#val/test
    OtherUp_Graph3,edgeIndexg1,edgeIndexg2,edgeIndexg3 =GraphFusion(g2_edgesweight,txtsim3,dataloader, device, model,edgeindex,edgeindex_values,datalabels,features,alltraincates) #此时的g3可能为val_g3或者test_g3
#     optimizer.step()  #更新DFGAT模型 
#     Graph3,Graph3_features,graphembeddings=GraphFusion(dataloader, device, model1,edgeindex,edgeindex_values,datalabels) #此时的g3可能为val_g3或者test_g3
#   #j计算GSL loss1--vaL/test图的GSL损失
#     print(Stop)--传入lab2/lab3，以测试val集分类acc或test分类acc
#     optimizer.zero_grad()
# dgl._ffi.base.DGLError: There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, causing silent performance regression. Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run.
#     Graph3=dgl.add_self_loop(Up_Graph3) #gat需要添加图的自循环
    # print( OtherUp_Graph3,edgeIndexg1,edgeIndexg2,edgeIndexg3,dataloader)
#     Graph(num_nodes=1000, num_edges=1391,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={}) tensor([[  0,   0,   0,  ..., 997, 998, 999],
#         [  0, 667,  23,  ..., 997, 998, 999]]) tensor([[  0,   0,   0,  ..., 997, 998, 999],
#         [  7, 160,   0,  ..., 997, 998, 999]]) tensor([[  0,   0,   0,  ..., 997, 998, 999],
#         [  7, 160,   0,  ..., 997, 998, 999]]) <dgl.dataloading.dataloader.GraphDataLoader object at 0x0000011FF176AD68>
#     print(Stop)
    UpdatedEvaluateG3,update_Evaluateg3features=FeaFusion_GCL(dataloader,edgeIndexg1,edgeIndexg2,edgeIndexg3,alltraincates,other_p)
    # print(" ValTestGCL_loss: ", ValTestGCL_loss)
#     GATs require self-loops to work. The self-loops themselves are not part of the training/validation/test set.
#  My suggestion would be adding self-loops to the training/validation/test graph first, then compute the scores on the edges for the edges that are not the added self-loops.
    score,precision, recall, F1,logits_other= evaluate( UpdatedEvaluateG3,update_Evaluateg3features, datalabels,onehotlabs, model) #传入变换的300维特征  调用evaluate函数：  传入的是分类模型model2， 单个图评估调用这个---传入融合的val_g3,val_g3的标签（不再是val_g1/g2)
    loss22 = loss_fcn(logits_other.to(device), datalabels.to(device))   #输出值logits_other和真实的datalabels来计算交叉熵损失， logits2 = model(train_g3, traing3_features,GFusion)
#     loss2.backward()
    totalloss_others=loss22
#     optimizer.zero_grad() #val/test清空过往梯度
#     optimizer.zero_grad() #分类模型梯度清零，清空过往梯度
#     totalloss_others.backward()
#     optimizer.step()  #DFGAT分类时更新模型
#     total_score += score #所有val图得分求和
    # print("score: ",score)  
#     totalloss=loss1.item()+loss2.item()
    print("totalloss:",totalloss_others)
#     print(stop)
    return score,precision, recall, F1,loss22.item()  # return average score val/test loss

# 如果你的是多分类（or多标签2分类），你可以将你的损失函数改为BCEWithLogitsLoss
# pred:  [[1 0 0 1]
#  [0 0 1 0]
#  [1 0 0 1]]---c封装不需要输入 [ 0 0 0 0 1]，只需要输入 4 就行---函数内部会自己处理成 one hot 格式
# loss1:  tensor([[3.1922]], grad_fn=<LogBackward0>)
# traing3_features shape:  torch.Size([80, 300])
# loss_fcn:  CrossEntropyLoss()
# loss2.item():  1.389633297920227
# loss2:  tensor(1.3896, grad_fn=<NllLossBackward0>)
#    train(train_dataloader, val_dataloader, device, model1) 
def GraphFusion_Train(train_dataloader, val_dataloader, device, model,features):
    #分两大步1执行：model1图融合和模model2型训练
#每次DFGAT模型执行一次完整操作（图融合+训练/val/test操作，即“计算一次梯度，更新一次模型”
    optimizer= torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=WD) #loss 太大-》sgd？
#     optimizer= torch.optim.SGD(model.parameters(), lr=5e-3, weight_decay=0)
#     for epoch in range(2000):  #E3000-正式的train、val与test可以设置更大些的epoch，如2000/3000+GPU上---更高的acc
    epochs=[]
    allaccs=[]
    allprecisions=[]
    allrecalls=[]
    allF1s=[]
    for epoch in range(500):  #E3000-正式的train、val与test可以设置更大些的epoch，如2000/3000+GPU上---更高的acc
#         train_g3, loss1,updatedg3features =GraphFusion(traing2_edgesweight,traintextsim,train_dataloader, device, model,edge_index,edge_index_values,lab,features,alltraincates) #默认传入model1,train_g1/g2的标签lab 
        #1-g1g2+GAT编码-》crossattention:g3；2-g1g2+GCN编码+g3的新嵌入-》GCL，3-更新g3嵌入+生成动态图结构--》分类。。。----》利用g1g2的嵌入-》GAT编码
        train_g3,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3  =GraphFusion(traing2_edgesweight,traintextsim,train_dataloader, device, model,edge_index,edge_index_values,lab,features,alltraincates)
        # train_g3,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3  
        # =GraphFusion(traing2_edgesweight,train_textsim[ii],train_dataloader, device, model,edge_index,edge_index_values,lab,features,alltraincates) #默认传入model1,train_g1/g2的标签lab 

#     print("updated graph_3.ndata['feat']:" ,graph_3.ndata['feat'])  #updated graph_3.ndata['feat']: tensor([[ 0.7889,  7.4189, -0.8719,  ..., -1.4548,  8.8519, -0.9236],
#     print(Stop)
#     print("loss1: ",loss1) train每个epoch一次特征融合+GCL；val/test每5个epoch进行一次
    
        UpdatedG3,update_g3features=FeaFusion_GCL(train_dataloader,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3,alltraincates,train_p)
        # print(" GCL_loss: ", GCL_loss) #loss1:  tensor([[-3.1890]], grad_fn=<NegBackward0>)
#         print(stop)
#         loss2=model2.train()  GraphFusion(g2_edgesweight,txtsim3,dataloader
        
#     loss1.backward() #loss1.backward(),loss2.backward(
#         optimizer.step() 
#         optimizer.zero_grad()
        loss2,loss_fcn=train(UpdatedG3,update_g3features) #传入g1，g2,原史特证， 传入变换的300维特征，loss2-分类损失，
#         loss2,loss_fcn=train(train_g3, traing3trans_features) #传入变换的300维特征，loss2-分类损失，
#         print("loss2: ",loss2)
        #此时为lamb*GCL_loss+分类loss
        total_loss=loss2  #  loss--张量对象:  tensor(0.0001, grad_fn=<NllLossBackward0>)
# loss.item()--张量的值:  0.00010086882684845477  total_loss=param['eta']*loss1+(1-param['eta'])*loss2 #eta:0.8 # loss:  tensor(9.7712, grad_fn=<MeanBackward0>)
        optimizer.zero_grad() #分类模型梯度清零，清空过往梯度
        total_loss.backward() #反向传播，计算当前梯度 AttributeError: 'float' object has no attribute 'backward'
        optimizer.step()  #根据梯度更新分类模型网络参数
        # print("lamb:",lamb)
        # print("GCL_loss.item():",GCL_loss.item())
        # print("loss2.item():",loss2.item())
        print(
          "Epoch {:05d} |loss2.item(){:.4f} | total loss {:.4f} |".format(
            epoch,loss2.item(), loss2.item()
         )
#         print(
#             "Epoch {:05d} | Loss {:.4f} |".format(
#                 epoch, total_loss / (batch_id + 1)
#             )
       )
        #验证的图，传入val_g3 
        if (epoch + 1) % 5 == 0: #每5轮评估一次(val一次，test一次）  evaluate_in_batches(g2_edgesweight,
            val_score,precision, recall, F1,totalval_loss = evaluate_in_batches( #只有一个val图3， 评估时，test的2个图求平均分 TypeError: argmax(): argument 'input' (position 1) must be Tensor, not tuple
    
                valg2_edgesweight,othertextsim,val_dataloader, device, model,edge_index2,edge_index_values2,lab2,valonehotlabs, loss_fcn,features
            )  # evaluate F1-score instead of loss,loss2,loss_fcn
            print(
                "                            Val Acc.  {:.4f},precision.  {:.4f}, recall.  {:.4f}, F1.  {:.4f}  ".format(
                    val_score,precision, recall, F1
                )
            )
            print("Validation total loss: ",totalval_loss)
            print("Testing...")
            test_score, precision,recall,F1,totaltest_loss = evaluate_in_batches(
                testg2_edgesweight,othertextsim,test_dataloader, device, model,edge_index3,edge_index_values3,lab3,testonehotlabs,loss_fcn,features
            )
            print(
                "                       Test Acc.  {:.4f}, precision.  {:.4f}, recall.  {:.4f}, F1.  {:.4f}  ".format(
                   test_score,precision, recall, F1
                ))
            print("Test total loss: ",totaltest_loss)
            epochs.append(epoch)
            allaccs.append(test_score)
            allprecisions.append(precision)
            allrecalls.append(recall)
            allF1s.append(F1)
            print("Test loss: ",totaltest_loss)
    i=0
    testmax=allaccs[i]
    for i in range(len(allaccs)):
        if testmax<=allaccs[i]:
            testmax=allaccs[i]
    maxind=0
    for m in range(len(allaccs)):
        if allaccs[m]==testmax:
            print(m)
            maxind=m
    print("Testbestacc:",testmax)
    print(
           "Test best Acc.  {:.4f}, precision.  {:.4f}, recall.  {:.4f}, F1.  {:.4f}  ".format(
                testmax,allprecisions[maxind], allrecalls[maxind], allF1s[maxind]
        ))

#也可以"复用IDEC论文-Improved Deep Embedded Clustering with Local Structure Preservation, IJCAI 2017中的重构损失"（即，输入输出之间的损失最小，误差最小）
#  reconstr_loss = F.mse_loss(x_bar, x) #torch.nn间的functional函数-》输入的数据嵌入x，x输出的嵌入x_bar===>类似GAN/扩散模型对抗的训练，“重构输入输出的误差，误差最小”
#             kl_loss = F.kl_div(q.log(), p[idx.type(torch.long)])
# #将x_bar和x类比于z(g1),z(g3）或z(g2),z(g3---三种图的嵌入,重构g1和g3，g2和g3的图误差==》“复用重构损失”+自己场景，“无需引用，直接使用mse_loss来计算即可”
# 论文公式(8): zi = fW(xi) and fW and gW0 are encoder and decoder
# mappings respectively
#             print("kl_loss:",kl_loss)
#             print("reconstr_loss:",reconstr_loss) #“重构输入输出之间的损失”
def ComputeGSLloss(z1: torch.Tensor, z2: torch.Tensor,z3: torch.Tensor):
#！！！注意，计算sim之前，可以先变换z1,z2，z3嵌入+“一个两层MLP层”【类似GCA对比目标】----》投影头变换 a nonlinear projection to enhance the expression power of the critic function
#     loss13=torch.exp(sim(allgraphembeddings[0],allgraphembeddings[2])/tau)
#     loss23=torch.exp(sim(allgraphembeddings[1],allgraphembeddings[2])/tau)
    loss13=torch.exp(sim(z1,z3)/tau)
    loss23=torch.exp(sim(z2,z3)/tau) #可以+两层的MLP（即，一个Project head转换图嵌入，再计算sim）
#     print("loss13:{},loss23:{}".format(loss13, loss23)) #loss13:tensor([[12.1550]], grad_fn=<ExpBackward0>),loss23:tensor([[12.1089]], grad_fn=<ExpBackward0>)
#     loss1=-torch.log(loss13+loss23)  #sim越来越小--》log越来越小？
    print(loss13,loss23) #tensor([[12.1703]], grad_fn=<ExpBackward0>) tensor([[12.1592]], grad_fn=<ExpBackward0>)
#     loss1=torch.log(loss13+loss23)  #sim越来越小--》log越来越小？ torch.log:e为底，tor.log10()以底数10为底
#     print(loss1) #torch.log(24.32)=3.19,torch.log10(24.32)=1.38    tensor([[3.1915]], grad_fn=<LogBackward0>)
# 1，c多学习同类优秀顶会论文，如GCA,GraphCL等---》"深度揣摩、探究某个点"篇【如对比loss函数仔细分析~论文公式+论文代码”<---->"类比+关联到自己的场景中"
# ===>2，“改进或修正loss函数”(使之更符合loss函数值的变化规律，loss变换趋势等）-利用对称性，e^(sim(z1,z3)/tau)/((e^(sim(z1,z3)/tau)+e^(sim(z2,z3)/tau))+e^(sim(z2,z3)/tau)/(e^(sim(z1,z3)/tau))+e^(sim(z2,z3)/tau))
# ----3,c模仿GCA的semi-loss（g1图和g3图的loss/g1，g3图+g2，g3图的总loss+g2图和g3图的loss/g1，g3图+g2，g3图的总loss===》取loge的对数，接着取反
#    -torch.log(item1)+(-torch.log(item2))=-[torch.log(item1)+torch.log(item2)]
    fm=loss13+loss23
    item1=-torch.log(loss13/fm) #0<loss13/fm<1=>torch.log(loss13/fm)<0=<item1>0
    item2=-torch.log(loss23/fm)
    loss1=item1+item2
    print(item1,item2,loss1) #tensor([[0.6928]], grad_fn=<NegBackward0>) tensor([[0.6935]], grad_fn=<NegBackward0>) tensor([[1.3863]], grad_fn=<AddBackward0>)
#     print(Stop)
    return loss1
def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
# def GraphFusion(train_dataloader, device, model): #1.c先加载g1，g2进行图融合，得到新图g3
#！！！此时，应该为图融合+图对比学习两个模块
def GraphFusion(g2_edgesweight,txtsim3,dataloader, device, model,edgeindex,edgeindex_values,data_labels,features,alltraincates): #传入标签参数：train/val/test,  1.先加载g1，g2进行图融合，得到新图g3
#     optimizer1= torch.optim.Adam(model1.parameters(), lr=5e-3, weight_decay=0) #model1和model2可以共用lr和weight_decay参数，优化器：模型参数，学习率和权重衰减等
#     model1.GraphFusion() 
    GFusion=True
    edgeatts=[]
    nodeembedds=[]
    NodeIms=[]
    graphembeddings=[]
    g1g2Edges=[]
    HiddenEmbeds=[]
    for batch_id, batched_graph in enumerate(dataloader): #train时为traindataloader，val时为valdataloader，  KeyError: 0，一个图一个图test
        batched_graph = batched_graph.to(device)
#     g.node[ind]['type'] = cate #根据下标给每篇短文本的节点添加type属性值，到图g中
#       edata_schemes={}) 取出batched_graph中的当前图的特征和标签
        features = batched_graph.ndata["feat"]#获取批中当前图的特征 ， RuntimeError: expected scalar type Double but found Float
#             features = batched_graph.ndata["feat"].float64()
        labels = batched_graph.ndata["label"].long() #RuntimeError: expected scalar type Long but found Int
#             labels = batched_graph.ndata["label"].float() #RuntimeError: expected scalar type Long but found Float
   #先图融合，得到新图,GSL计算图融合loss，接着使用融合的新图g3-train模型+分类 +g1、g2图的嵌入
        edge_attention,node_embeddings,HiddenEmbeds,Crossattg3=model(batched_graph,features,GFusion, HiddenEmbeds) #调用forward函数，传入GFusion标记
        edgeatts.append(edge_attention)

       
        g1g2Edges.append(batched_graph.edges())
#   batched_graph.edges():  (tensor([  0,   0,   0,  ..., 122, 122, 122]), tensor([  0,  15, 119,  ...,  66,  18, 122]))   batched_graph:  Graph(num_nodes=140, num_edges=140,
#       ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float64), 'label': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={}) batched_graph.edges():  (tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13, 138, 139]), tensor([  0,   1,   2,   3,   4, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139]))
#         print(edge_attention,node_embeddings,NodeImports)
        # print(edge_attention.shape) #torch.Size([1000, 1]) torch.Size([140, 1]) torch.Size([140, 1024]) 140 torch.Size([1971, 1]) torch.Size([80, 1024]) 80
   
    graph_3, features =Construct_Graph3(g2_edgesweight,edgeatts,edgeindex,edgeindex_values,features,txtsim3)#返回的g3特征应该是隐层的1024维特征（和g1，g2的相同）
#     read3=AvgReadout() #创建一个实例化对象--实例化AvgReadout类--默认调用def forward(self, seq, msk)方法：计算图嵌入
#     msk3=None 
#     graph_3features=graph_3_features.unsqueeze(0)  #第1个维度扩展：二维-》三维
#     graph_embedding3 = read3(graph_3features, msk3)   #class AvgReadout(nn.Module):  传入参数h和msk实例化self.read函数
#     print(graph_embedding3.shape,graph_embedding3) #图g3的嵌入--注意train/val/test都需要计算
#     graphembeddings.append(graph_embedding3)
#     print(graph_embedding3.shape,graph_embedding3) #g3图嵌入
    graph_3.ndata["label"]=data_labels.to(device)  #train_g3与train图g1，g2共用相同label--main函数有这个变量，直接调用
#     print("graph_3: ", graph_3) 
 
    edgeIndex_g1=[]
    edgeIndex_g2=[]
    edgeIndex_g3=[]
#     print(g1g2Edges[0],g1g2Edges[1]) #(tensor([  0,   0,   0,  ..., 122, 122, 122]), tensor([  0,  15, 119,  ...,  66,  18, 122])) (tensor([  0,   1,   2,   3,   4,  
#     print(g1g2Edges[0][0],g1g2Edges[0][1]) #tensor([  0,   0,   0,  ..., 122, 122, 122]) tensor([  0,  15, 119,  ...,  66,  18, 122])
    g1Edges_row=g1g2Edges[0][0].tolist()
    g1Edges_col=g1g2Edges[0][1].tolist()
    edgeIndex_g1.append(g1Edges_row)
    edgeIndex_g1.append(g1Edges_col)
    edgeIndex_g1=torch.tensor(edgeIndex_g1)
#     print(edgeIndex_g1) # tensor([[  0,   0,   0,  ..., 122, 122, 122], [  0,  15, 119,  ...,  66,  18, 122]])
    g2Edges_row=g1g2Edges[1][0].tolist()
    g2Edges_col=g1g2Edges[1][1].tolist()
    edgeIndex_g2.append(g2Edges_row)
    edgeIndex_g2.append(g2Edges_col)
    edgeIndex_g2=torch.tensor(edgeIndex_g2)
#     print(edgeIndex_g2) #tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13, 34, 135, 136, 137, 138, 139]])
#     print("graph_3.edges():",graph_3.edges()) #graph_3.edges(): (tensor([  0,   0,   0,  ..., 139, 139, 139]), tensor([ 0,  1,  2,  ..., 27, 28, 29]))
#     print(graph_3.edges()[0],graph_3.edges()[1]) #tensor([  0,   0,   0,  ..., 139, 139, 139]) tensor([ 0,  1,  2,  ..., 27, 28, 29])
#     print(graph_3.edges()[0].shape,graph_3.edges()[1].shape) #torch.Size([4200]) torch.Size([4200])
    g3Edges_row=graph_3.edges()[0].tolist()
    g3Edges_col=graph_3.edges()[1].tolist()
    edgeIndex_g3.append(g3Edges_row)
    edgeIndex_g3.append(g3Edges_col)
    edgeIndex_g3=torch.tensor(edgeIndex_g3)
#     print("g3 numedges:")
#     g3 numedges:由于取g1和g2的并集，所以g3的边实际为g1中的边
# tensor([[ 0,  0,  0,  ..., 20, 20, 20],
#         [ 0,  1,  2,  ..., 37, 38, 39]]) torch.Size([1312]) torch.Size([1312])
#         g3 numedges:
# tensor([[  0,   1,   2,  ..., 994, 997, 999],
#         [  0,   1,   2,  ..., 994, 997, 999]]) torch.Size([6408]) torch.Size([6408])
#         g3 numedges:
# tensor([[   0,    0,    0,  ..., 9610, 9611, 9617],
#         [   0, 1648, 2105,  ..., 9610, 9611, 9617]]) torch.Size([494580]) torch.Size([494580])
    # print(edgeIndex_g1,edgeIndex_g1[0].shape,edgeIndex_g1[1].shape) #tensor([[ 0,  0,  0,  ..., 18, 18, 18],  [ 0,  1, 34,  ..., 20, 13, 18]]) torch.Size([576]) torch.Size([576])
    # print(edgeIndex_g2,edgeIndex_g2[0].shape,edgeIndex_g2[1].shape) #       33,  2, 11, 13, 18, 30, 35, 36, 37, 39, 34,  4, 11, 18, 32, 36, 37, 38]]) torch.Size([414]) torch.Size([414])
    # print(edgeIndex_g3,edgeIndex_g3[0].shape,edgeIndex_g3[1].shape) #tensor([[  0,   0,   0,  ..., 139, 139, 139],
#         [  0,   1,   2,  ...,  27,  28,  29]])
#    tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,33,  2, 11, 13, 18, 30, 35, 36, 37, 39, 34,  4, 11, 18, 32, 36, 37, 38]]) torch.Size([414]) torch.Size([414])
#    tensor([[   0,    0,    0,  ..., 6617, 6621, 6622],
#         [   0,  611,  675,  ..., 6617, 6621, 6622]]) torch.Size([42288]) torch.Size([42288])
# tensor([[   0,    0, 1548,  ..., 6955, 6958, 6959],
#         [1548,    0,  166,  ..., 6955, 6958, 6959]]) torch.Size([27991]) torch.Size([27991])
# tensor([[   0,    0, 1548,  ..., 6955, 6958, 6959],
#         [1548,    0,  166,  ..., 6955, 6958, 6959]]) torch.Size([27991]) torch.Size([27991])
# random drop g3 edges...
# tensor([[1548, 1548, 1548,  ..., 6950, 6955, 6958],
#         [ 166,  288,  459,  ..., 6950, 6955, 6958]]) torch.Size([22297]) torch.Size([22297]) None
    
    return graph_3,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3 #返回g1/g2/g3的图嵌入
def FeaFusion_GCL(dataloader,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3,alltraincates,p_removeedge):
    GFusion=True
    HiddenEmbeds=[]
    for batch_id, batched_graph in enumerate(dataloader): #train时为traindataloader，val时为valdataloader，  KeyError: 0，一个图一个图test
        batched_graph = batched_graph.to(device)
    #     g.node[ind]['type'] = cate #根据下标给每篇短文本的节点添加type属性值，到图g中
    #       edata_schemes={}) 取出batched_graph中的当前图的特征和标签
        features = batched_graph.ndata["feat"]#获取批中当前图的特征 ， RuntimeError: expected scalar type Double but found Float
    #             features = batched_graph.ndata["feat"].float64()
        labels = batched_graph.ndata["label"].long() #RuntimeError: expected scalar type Long but found Int
    #             labels = batched_graph.ndata["label"].float() #RuntimeError: expected scalar type Long but found Float
       #不需要图融合，只需要计算crossattention即可
        edge_attention,node_embeddings,HiddenEmbeds,Crossattg3=model(batched_graph, batched_graph.ndata["feat"],GFusion, HiddenEmbeds) 
#         print(Crossattg3[0],Crossattg3[0].shape) #torch.Size([40, 384])
#         print(Stop)
        if batch_id==1: #g1g2，GCL-GCN更新嵌入---》GAT编码=》重新crossatte--->g3新嵌入+g1g2GCL====》生成更更新的g3嵌入----分类
            z1 = GCL_model(batched_graph.ndata["feat"].to(torch.float32), edgeIndex_g1.to(device))#传入节点特征和边， 调用GCLmodel（图卷积）分别对2个增强的图视图进行编码、投影--》图对比学习
            batched_graph.ndata["feat"]=z1
    #     x_1:  tensor([[0., 0., 0.,  ..., 0., 0., 0.], tensor([[    0,     0,     0,  ..., 18331, 18331, 18331],
    #         [ 5111, 12716, 12963,  ..., 13985, 14816, 17748]])
        else:
            z2 = GCL_model(batched_graph.ndata["feat"].to(torch.float32), edgeIndex_g2.to(device))  #g2图特征
            batched_graph.ndata["feat"]=z2 #更新train_g2图特征

    # print("random drop g3 edges...") #random drop g3 edges...需要edgeIndex_g3参数，
    Update_edgeindexg3, edge_attrg3=dropout_adj(edgeIndex_g3, p=p_removeedge)
    # Update_edgeindexg3, edge_attrg3=dropout_adj(edgeIndex_g3, p=0.2) #复用torcch_geometric.utils--drop_adj代码，train:0.1--414*0.1=41.4----414-41.4=373
    # print(Update_edgeindexg3,Update_edgeindexg3[0].shape,Update_edgeindexg3[1].shape, edge_attrg3) # 11, 13, 18, 35, 36, 37, 39, 34, 11, 18, 32, 36, 37, 38]]) torch.Size([374]) torch.Size([374]) None
    #     print(stop)
#     features=features.to(torch.float32) #RuntimeError: expected scalar type Double but found Float
    #     print(features,features.shape) dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got 6960 and 6959 instead.
      
    Updated_G3 = dgl.graph((Update_edgeindexg3[0], Update_edgeindexg3[1])) #边的行，列直接构造dgl图(不需要networkx图，tuple复杂代码）
    # print(Updated_G3.nodes()) #打印g3图中所有节点 tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    #         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    Update_edgeindexg3row=Update_edgeindexg3[0].tolist()#行
    Update_edgeindexg3col=Update_edgeindexg3[1].tolist()#列
    UpdatedG3Nodes=Updated_G3.nodes().tolist()
    # print(UpdatedG3Nodes) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    #     print(Crossattg3[0],Crossattg3[0].shape) 
    AddallNodes_G3EdgeIndex=[]
    AddallNodes_G3EdgeIndex.append(Update_edgeindexg3row)
    AddallNodes_G3EdgeIndex.append(Update_edgeindexg3col)
#     print(z1.shape[0]) #40
#     print(Stop)
    for i in range(z1.shape[0]): #在当前fea所有节点查找当前更新的图元素
    #         print(torch.tensor(i)) #tensor(39)
    #         print(i)
        if i not in UpdatedG3Nodes: 
            # print("no exist..")
            # print(i)
            Update_edgeindexg3row.append(i) 
            Update_edgeindexg3col.append(i)
            AddallNodes_G3EdgeIndex[0].append(i)
            AddallNodes_G3EdgeIndex[1].append(i)
    UpdatedG3 = dgl.graph((torch.tensor(Update_edgeindexg3row), torch.tensor(Update_edgeindexg3col))).to(device) #加入缺少的节点后重新构图
    UpdatedG3.ndata['feat']=Crossattg3[0].to(device)
    #     print(torch.tensor(Update_edgeindexg3row)) #tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1
    #     print(stop)
  

    #调用训练好的GCN模型对融合的g3图数据和图结构进行编码（仍传入原始特征+图结构-"更新的GCN模型结构上"===》得到优化的、更新的g3特征-仍为384维
    
# graph_3.ndata['feat']: tensor([[-0.0068,  0.0092, -0.0269,  ...,  0.0242,  0.0080, -0.0267],
     #添加自循环
    UpdatedG3=dgl.add_self_loop(UpdatedG3) #dgl._ffi.base.DGLError: There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, causing silent performance regression. Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run.
    # print(UpdatedG3)
    return UpdatedG3,UpdatedG3.ndata['feat']
#     return graph_3, g3trans_features,GCL_loss,update_g3features #返回g1/g2/g3的图嵌入
# def train(train_dataloader, val_dataloader, device, model): #GraphFusionandTrain
def train(train_g3, traing3_features): #GraphFusion和train分别优化各自模块？
    # define loss function and optimizer
    # print("traing3_features shape: ",traing3_features.shape) #traing3_features shape:  torch.Size([80, 300])
    
    GFusion=False
#     model = DFGAT(1024, 256, out_size, heads=[4, 4, 6],GFusion=True).to(device) #GCN每层的multi-head数分别为4,4,6
#融合模型輸入1024个神经元，分类模型输入300个神经元-----两个模块的模型结构不同 （不能共用一个model？）
#可以把graph3的1024feature->300维（线性变换*W或者PCA降维
#     loss_fcn=nn.MSELoss() ppi数据集-inductive设置的 multi-label分类
 #loss函数，优化器是GAT模型（图融合模块，分类模块）的公共部分
    loss_fcn=nn.CrossEntropyLoss() #交叉熵分类损失函数
#需要把HGAT数据集中的networkx格式的graph--》dgl格式的graph（使数据集中的格式和dataset[0]完全保持一致）
#     loss_fcn=nn.NLLLoss()
    # print("loss_fcn: ",loss_fcn)  # loss_fcn:  CrossEntropyLoss()
#轮数表示模型学习的总次数，，每一轮：g1，g2图融合成g3+将g3输入到模型继续分类
#     for epoch in range(2000):  #E3000-正式的train、val与test可以设置更大些的epoch，如2000/3000+GPU上---更高的acc
#     model2.train() #model1为融合的模型对象， 正式开始，一轮轮训练， 启用BatchNormalization和Dropout， 调用train函数，正式开始训练
#注意，train模型前，先进行图融合，即g1+g2==>g3,然后进行GSL，优化g3+同时“进行分类”
# optimizer:  Adam (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     eps: 1e-08
#     lr: 0.005
#     weight_decay: 0
# )
    logits2 = []
#     total_loss = 0
        # mini-batch loop，  先实现图g1和图g2融合--》再使用融合后的图g3进行分类
#     print("train_dataloader:",train_dataloader) #train_dataloader: <dgl.dataloading.dataloader.GraphDataLoader object at 0x00000228299741D0>
# TypeError: forward() missing 1 required positional argument: 'HiddenEmbeds'
    HiddenEmbeds=[]
#     model2.train() #调用正常GAT模型训练 dgl._ffi.base.DGLError: There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, causing silent performance regression. Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run.
    logits2 = model(train_g3, traing3_features,GFusion,HiddenEmbeds) #传入GFusion参数，设置此时DFGAT模型不需要融合， 需要用到traing3的特征，
#     print("logits2:",logits2) 
#         logits = model(batched_graph, features)  #model2分类，传入新的g3图和对应的g3图节点特征，  logits《--》前向传播forward中输出层的h值？
#     print(logits2.shape,logits2)
#             loss = loss_fcn(logits[train_mask], labels[train_mask])
    loss2 = loss_fcn(logits2.to(device), lab.to(device)) #损失函数  loss_fcn=nn.CrossEntropyLoss()
#     print("loss2: ",loss2) #loss2:  tensor(1.3863, grad_fn=<NllLossBackward0>)
    # print("loss2.item(): ",loss2.item()) #loss2.item():  1.3862965106964111
#train中独立的反向传播，优化，loss2
#     loss2.backward()
    return loss2,loss_fcn
#！！！边调试边写》》提高编码效率....
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
   #添加自循环
    g2=dgl.add_self_loop(g2)
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
    dataset='snippets'
    # lujing="agnews-mymethod"
#     dataset='twitter'
#     sys.stdout = Logger(logdir + "{}-TrainValTestFusionOnceRemoveMLPmodule_pretest1230.log".format(dataset))
    sys.stdout = Logger(logdir + "{}-MSGCL-D2FSF_Onlyg3Ablation.log".format(dataset)) #1230应该是融合所有的
    path2 = './data/'
    path = path2+'{}'.format(dataset)  # label = g.ndata['label']
#     sys.stdout = Logger(logdir + "{}-FusionOnceeNeighg1∩g2-g3Dynamictrainothersim-totallossxiuzheng-addCrossAtt+GCLmodule_OnlyTrainBP-TrainSupervisedGCL+UpdatePara.log".format(dataset))
#     cate=2

    numnodes=12340 #当前节点总数
    out_size = 8#分类的类别数
    current_labels=[0,1,2,3,4,5,6,7]
    tau=0.4 #GSL loss中的tau值设置 GSL loss1-->(-logxx)
    lamb=0.8 #损失系数，L=L1+lamb*L2 (L1为分类loss，L2为GSL loss）
#     textsim1=0.1 #train图，g2的边相似度
    traintextsim=0.1
    othertextsim=0.1
    learningrate=0.0035
    traing2_textsim=0.6  #必须设置为固定值0.6--保持g1，g2图的差异
    # trainp=[0.6]
    # otherp=[0.55]
    # WDs=[0.06]
    train_p=0.3
    other_p=0.45
    WD=0.0015
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
    train_dataset_list.append(train_g1)
    val_dataset_list.append(val_g1)
    #同时添加g1，g2图
    train_dataset_list.append(train_g2) #换成g2图单独测试
    val_dataset_list.append(val_g2)
    print("train_dataset_list:",train_dataset_list)
    train_dataloader = GraphDataLoader(train_dataset_list, batch_size=1) #train图20个，每批2个图--20/2=10批
    val_dataloader = GraphDataLoader(val_dataset_list, batch_size=1)
    print("train_dataloader: ",train_dataloader) #train_dataloader:  <dgl.dataloading.dataloader.GraphDataLoader object at 0x000002C3DD976F28>
            #     print(stop)
    #4.t图融合阶段：节点融合和边融合--只需要输入层和隐层两层-----此时已经实例化，+“真实的g1-train,val,test"
                #图融合和模型train learning共用一个GAT模型
    print("GraphFusion and Training...")
            #     gfusion = GraphFusion(in_size, 256, heads=[4, 4]).to(device) #GCN每层的multi-head数分别为4,4,6
            #     print(Stop)
                #5，训练模型  ，每隔5轮验证一次，测试一次模型
            #     train(train_dataloader, val_dataloader, device, model2) #RuntimeError: mat1 and mat2 shapes cannot be multiplied (8040x384 and 50x1024)
            #     print("train_dataloader:",train_dataloader) #train_dataloader: <dgl.dataloading.dataloader.GraphDataLoader object at 0x0000025DFF8E0BE0>
            #     print("val_dataloader： ",val_dataloader) #val_dataloader：  <dgl.dataloading.dataloader.GraphDataLoader object at 0x0000025DD094A7F0>
            #     print(STOP)
                #5，测试模型 test the model
            #     print("Testing...")
            #     test_dataloader = GraphDataLoader(test_dataset, batch_size=2) #2个测试图
    #只用1个图测试
    test_dataset_list=[]
    test_dataset_list.append(test_g1)
    #同时添加g1和g2图
    test_dataset_list.append(test_g2) #g2图单独test
            #     test_dataset_list.append(test_dataset[0])
            #     test_dataloader = DataLoader(data, batch_size=1)
            #     test_dataloader = GraphDataLoader(data, batch_size=1)
    test_dataloader = GraphDataLoader(test_dataset_list, batch_size=1) #2c个测试图
    #调用train函数，GAT模型开始训练并隔一定epoch，开始val，test
    GraphFusion_Train(train_dataloader, val_dataloader, device, model,train_g1.ndata["feat"]) 
            #     GraphFusion_Train(train_dataloader, val_dataloader, device, model) 
#     GraphFusion_Train(train_dataloader, val_dataloader, device, model1) 
    #测试test，可以写在val后面（即，同样地，每隔5轮测试一次。。。。）
    # test_score = evaluate_in_batches(test_dataloader, device, model,edge_index3,edge_index_values3,lab3)
#     test_score = evaluate_in_batches(test_dataloader, device, model2,edge_index3,edge_index_values3,lab3)
#     avg_score = evaluate_in_batches(test_dataloader, device, model) #
#     print("test_dataloader:",test_dataloader)
    # print("test_score： ",test_score)
    # print("Test Accuracy {:.4f}".format(test_score))
    endtime=time.time() 
    print("cost time:",endtime-starttime)
#     