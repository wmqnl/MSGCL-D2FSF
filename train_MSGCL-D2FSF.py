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
from GCLModel import Encoder, MYGCL#
from GCLutils import get_base_model, get_activation
from crossattention import CrossAttention
# from projectLayer import ProjectLayer #导入MLP
from sklearn.metrics import precision_recall_fscore_support #P,R,F1新指标
from sklearn.metrics import accuracy_score
from copy import deepcopy
import time
starttime=time.time() 
logdir = "log/"

class MSGCLD2FSF(nn.Module): 
    def __init__(self, in_size, hid_size, out_size, heads): #attention中的multi-head注意力
        super().__init__()
        self.gat_layers = nn.ModuleList()
#         self.GFusion=GFusion #bool标记
        dim=384 #隐层维度，256*4
        self.crossatt=CrossAttention(dim) 
        dtemp = 32
        self.W = nn.Linear(dim, dtemp) 
        self.w = nn.Linear(dtemp, 1, bias=False) 
        # three-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu)
        )
        self.gat_layers.append( 
            dglnn.GATConv( #
                hid_size * heads[0],
                hid_size,
                heads[1], #隐层注意力头数
                residual=True,
                activation=F.elu,
            )
        )
        print("self.gat_layers: ",self.gat_layers)
        print( dglnn.GATConv(
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

    def forward(self, g, inputs,GFusion, HiddenEmbeds): 
        h = inputs.float()  
        for i, layer in enumerate(self.gat_layers):
            h, edge_attention = layer(g, h) 
#             h = layer(g, h)
            if i == 2:  
                h = h.mean(1)
            else:  # other layer(s)--其它层，拼接不同muli-head注意力计算的嵌入
                h = h.flatten(1)
                edge_attention=edge_attention.mean(1) 
                Crossattg3=[]
                if i==1 and GFusion==True: 
                    if len(HiddenEmbeds)==0: #当前g1
                        HiddenEmbeds.append(h) #g1
                    elif len(HiddenEmbeds)==1: #g2,开始使用crossatten融合
#                         print(h,h.shape,HiddenEmbeds[0],HiddenEmbeds[0].shape)
#                    
                        FusionAtt_features=self.crossatt(h.unsqueeze(1),HiddenEmbeds[0].unsqueeze(1)) #实例化一个对象必须和q，k，v的原始维度一样  RuntimeError: mat1 and mat2 shapes cannot be multiplied (40x1024 and 384x384)
#                    
                        CrossAtt_g3features=FusionAtt_features.squeeze(1)
                        Crossattg3.append(CrossAtt_g3features)
                    return edge_attention,h,HiddenEmbeds,Crossattg3 #隐层的g1，g2节点嵌入 AvgReadout-->g1,g2图嵌入 
        return h
    
   
def encode_onehot(dlabels):
    labelsls=dlabels.tolist()
    newlabels=[]
    for i in range(len(labelsls)):
        val=str(labelsls[i])
        newlabels.append(val)
    newlabelsarr=np.array(newlabels)

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

    HiddenEmbeds=[] #分类阶段，该参数不使用
    output = model(Up_g3, updatedg3features, GFusion,HiddenEmbeds)  #调用DFGAT的forward函数 ++loss2 (val/test loss）

    pred2 = torch.argmax(output, 1)
    [precision, recall, F1, support] = \
        precision_recall_fscore_support(y_true=dlabels.data, y_pred=pred2.cpu().detach().numpy(),labels=current_labels, average='macro')
    Accuracy= accuracy_score(dlabels.data, pred2.cpu().detach().numpy()) 
    return Accuracy,precision, recall, F1,output
#

def evaluate_in_batches(g2_edgesweight,txtsim3,dataloader, device, model,edgeindex,edgeindex_values,datalabels,onehotlabs, loss_fcn,features): #评估时只传入testdataloader
    totalloss=0
    alltraincates=[]#val/test
    OtherUp_Graph3,edgeIndexg1,edgeIndexg2,edgeIndexg3 =GraphFusion(g2_edgesweight,txtsim3,dataloader, device, model,edgeindex,edgeindex_values,datalabels,features,alltraincates) #此时的g3可能为val_g3或者test_g3

    ValTestGCL_loss,UpdatedEvaluateG3,update_Evaluateg3features=FeaFusion_GCL(dataloader,edgeIndexg1,edgeIndexg2,edgeIndexg3,alltraincates,other_p)
    score,precision, recall, F1,logits_other= evaluate( UpdatedEvaluateG3,update_Evaluateg3features, datalabels,onehotlabs, model) #传入变换的300维特征  调用evaluate函数：  传入的是分类模型model2， 单个图评估调用这个---传入融合的val_g3,val_g3的标签（不再是val_g1/g2)
    loss22 = loss_fcn(logits_other.to(device), datalabels.to(device))   #输出值logits_other和真实的datalabels来计算交叉熵损失， logits2 = model(train_g3, traing3_features,GFusion)
    totalloss_others=lamb*ValTestGCL_loss+loss22
    GCL_optimizer.zero_grad() #GCL优化器
    ValTestGCL_loss.backward() 
    GCL_optimizer.step()

    print("totalloss:",totalloss_others)
#     print(stop)
    return score,precision, recall, F1,lamb*ValTestGCL_loss.item()+loss22.item()  # return average score val/test loss

#
def GraphFusion_Train(train_dataloader, val_dataloader, device, model,features):
    
    optimizer= torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=WD) #loss 太大-》sgd？
#    
    for epoch in range(500):  
        train_g3,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3  =GraphFusion(traing2_edgesweight,traintextsim,train_dataloader, device, model,edge_index,edge_index_values,lab,features,alltraincates)
       
    
        GCL_loss,UpdatedG3,update_g3features=FeaFusion_GCL(train_dataloader,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3,alltraincates,train_p)
       
        loss2,loss_fcn=train(UpdatedG3,update_g3features)
#        
        total_loss=lamb* GCL_loss+loss2 
        GCL_optimizer.zero_grad() 
        optimizer.zero_grad() 
        total_loss.backward() 
        GCL_optimizer.step() 
        optimizer.step()  
        print(
          "Epoch {:05d} |GCL_loss.item() {:.4f} |loss2.item(){:.4f} | total loss {:.4f} |".format(
            epoch,GCL_loss.item(),loss2.item(), lamb*GCL_loss.item()+loss2.item()
         )

       )
        if (epoch + 1) % 5 == 0: 
            val_score,precision, recall, F1,totalval_loss = evaluate_in_batches( 
    
                valg2_edgesweight,othertextsim,val_dataloader, device, model,edge_index2,edge_index_values2,lab2,valonehotlabs, loss_fcn,features
            )  
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
            
def ComputeGSLloss(z1: torch.Tensor, z2: torch.Tensor,z3: torch.Tensor):

    loss13=torch.exp(sim(z1,z3)/tau)
    loss23=torch.exp(sim(z2,z3)/tau) 

    fm=loss13+loss23
    item1=-torch.log(loss13/fm) 
    item2=-torch.log(loss23/fm)
    loss1=item1+item2
    print(item1,item2,loss1) 
    return loss1
def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def GraphFusion(g2_edgesweight,txtsim3,dataloader, device, model,edgeindex,edgeindex_values,data_labels,features,alltraincates): #传入标签参数：train/val/test,  1.先加载g1，g2进行图融合，得到新图g3

    GFusion=True
    edgeatts=[]
    nodeembedds=[]
    NodeIms=[]
    graphembeddings=[]
    g1g2Edges=[]
    HiddenEmbeds=[]
    for batch_id, batched_graph in enumerate(dataloader):
        batched_graph = batched_graph.to(device)

        features = batched_graph.ndata["feat"]
        labels = batched_graph.ndata["label"].long() 
       
        edge_attention,node_embeddings,HiddenEmbeds,Crossattg3=model(batched_graph,features,GFusion, HiddenEmbeds) #调用forward函数，传入GFusion标记
        edgeatts.append(edge_attention)


        g1g2Edges.append(batched_graph.edges())
    graph_3, features =Construct_Graph3(g2_edgesweight,edgeatts,edgeindex,edgeindex_values,features,txtsim3)#返回的g3特征应该是隐层的1024维特征（和g1，g2的相同）

    graph_3.ndata["label"]=data_labels.to(device)  #train_g3与train图g1，g2共用相同label--main函数有这个变量，直接调用
 
    edgeIndex_g1=[]
    edgeIndex_g2=[]
    edgeIndex_g3=[]

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

    g3Edges_row=graph_3.edges()[0].tolist()
    g3Edges_col=graph_3.edges()[1].tolist()
    edgeIndex_g3.append(g3Edges_row)
    edgeIndex_g3.append(g3Edges_col)
    edgeIndex_g3=torch.tensor(edgeIndex_g3)

    
    return graph_3,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3 #返回g1/g2/g3的图嵌入
def FeaFusion_GCL(dataloader,edgeIndex_g1,edgeIndex_g2,edgeIndex_g3,alltraincates,p_removeedge):
    GFusion=True
    HiddenEmbeds=[]
    for batch_id, batched_graph in enumerate(dataloader): #train时为traindataloader，val时为valdataloader，  KeyError: 0，一个图一个图test
        batched_graph = batched_graph.to(device)
   
        features = batched_graph.ndata["feat"]
        labels = batched_graph.ndata["label"].long() 
        edge_attention,node_embeddings,HiddenEmbeds,Crossattg3=model(batched_graph, batched_graph.ndata["feat"],GFusion, HiddenEmbeds) 
#       
        if batch_id==1: 
            z1 = GCL_model(batched_graph.ndata["feat"].to(torch.float32), edgeIndex_g1.to(device))#传入节点特征和边， 调用GCLmodel（图卷积）分别对2个增强的图视图进行编码、投影--》图对比学习
            batched_graph.ndata["feat"]=z1
   
        else:
            z2 = GCL_model(batched_graph.ndata["feat"].to(torch.float32), edgeIndex_g2.to(device))  #g2图特征
            batched_graph.ndata["feat"]=z2 #更新train_g2图特征

    Update_edgeindexg3, edge_attrg3=dropout_adj(edgeIndex_g3, p=p_removeedge)

    Updated_G3 = dgl.graph((Update_edgeindexg3[0], Update_edgeindexg3[1])) #边的行，列直接构造dgl图(不需要networkx图，tuple复杂代码）
   
    Update_edgeindexg3row=Update_edgeindexg3[0].tolist()#行
    Update_edgeindexg3col=Update_edgeindexg3[1].tolist()#列
    UpdatedG3Nodes=Updated_G3.nodes().tolist()
   
    AddallNodes_G3EdgeIndex=[]
    AddallNodes_G3EdgeIndex.append(Update_edgeindexg3row)
    AddallNodes_G3EdgeIndex.append(Update_edgeindexg3col)
#
    for i in range(z1.shape[0]): 
        if i not in UpdatedG3Nodes: 
          
            Update_edgeindexg3row.append(i) 
            Update_edgeindexg3col.append(i)
            AddallNodes_G3EdgeIndex[0].append(i)
            AddallNodes_G3EdgeIndex[1].append(i)
    UpdatedG3 = dgl.graph((torch.tensor(Update_edgeindexg3row), torch.tensor(Update_edgeindexg3col))).to(device) #加入缺少的节点后重新构图
    UpdatedG3.ndata['feat']=Crossattg3[0].to(device)
 
    AddallNodes_G3EdgeIndex=torch.tensor(AddallNodes_G3EdgeIndex)
    
    z3 = GCL_model(Crossattg3[0],  AddallNodes_G3EdgeIndex.to(device)) 
   
    loss13 = GCL_model.loss(z1, z3,alltraincates, batch_size=None) #z1为g1图所有节点嵌入,z3为g3图所有节点嵌入
    loss23 = GCL_model.loss(z2, z3,alltraincates, batch_size=None)
    GCL_loss=1/2*(loss13+loss23)  #返回的有TrainSuperGCL，val和test的UnsuperGCL
    update_g3features = GCL_model(Crossattg3[0], AddallNodes_G3EdgeIndex.to(device))  #初始的Crossatten生成的g3嵌入+更新的GCN模型编码  第1次不更新GCL_model参数，直接使用原始features，第二次以后则使用更新参数的GCN模型编码
    
    UpdatedG3.ndata['feat']=update_g3features #更新g3图的节点特征
    UpdatedG3=dgl.add_self_loop(UpdatedG3) #dgl._ffi.base.DGLError: There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, causing silent performance regression. Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run.
    return GCL_loss,UpdatedG3,update_g3features

def train(train_g3, traing3_features): 
   
    
    GFusion=False

    loss_fcn=nn.CrossEntropyLoss() 

    logits2 = []
#    
    HiddenEmbeds=[]
    logits2 = model(train_g3, traing3_features,GFusion,HiddenEmbeds) 
#
    loss2 = loss_fcn(logits2.to(device), lab.to(device)) 
#   
    return loss2,loss_fcn
def Construct_Graph3(g2_edgesweight,edgeatts,edgeindex,edgeindex_values,features,txtsim3): 
  
        g1_edgeatt=edgeatts[0].tolist()
        g2_edgeatt=edgeatts[1].tolist()

        edgescores2=dict()
        count=0
        for key,value in g2_edgesweight.items():
            edgescores2[key]=g2_edgeatt[count][0] #
#            
            count=count+1
#     
        edgeindex1= edgeindex.tolist() 
        edgeindex_values1= edgeindex_values.tolist() 
        edgeweights1=dict() 
        for i in range(len(edgeindex1[0])):
            templs=[]
            templs.append(edgeindex1[0][i]) #对应行
            templs.append(edgeindex1[1][i]) #对应列
            temptuple=tuple(templs) 
#             
            edgeweights1[str(temptuple)]=edgeindex_values1[i] 
      
        edgescores1=dict()
        count1=0
        for key,value in edgeweights1.items():
            edgescores1[key]=g1_edgeatt[count1][0] #
#             
            count1=count1+1
        newedge_weights=dict() 
        w_greats=[] 

        g1updated_Edgescores=dict()
        g2updated_Edgescores=dict() 
        Normg1g2_Edgescores=[]
        g1UpdateEdScoresls=[]
        g2UpdateEdScoresls=[]
        g1g2_keys=[]
        for kk,vv in edgescores1.items(): 
            if kk in edgescores2: 
                g1g2_keys.append(kk)
#            
                edgeweights1[str(temptuple)]=edgeindex_values1[i] 
                g1updated_Edgescores[kk]=edgescores1[kk]
                g2updated_Edgescores[kk]=edgescores2[kk]
                g1UpdateEdScoresls.append(edgescores1[kk])
                g2UpdateEdScoresls.append(edgescores2[kk])
        Normg1g2_Edgescores.append(g1UpdateEdScoresls)
        Normg1g2_Edgescores.append(g2UpdateEdScoresls)
        Normg1g2_Edgescores=torch.Tensor(Normg1g2_Edgescores)
#         
        Normg1g2_Edgescores=F.softmax(Normg1g2_Edgescores,dim=0) 
#       
        NormallEdgeScoresls=Normg1g2_Edgescores.tolist()
#
        g1Norm_EdgeScores=dict()
        g2Norm_EdgeScores=dict()
        for x in range(len(NormallEdgeScoresls[0])):
            g1Norm_EdgeScores[g1g2_keys[x]]=NormallEdgeScoresls[0][x]
            g2Norm_EdgeScores[g1g2_keys[x]]=NormallEdgeScoresls[1][x]
# 
        g3_newedgeweights=dict()
        for kk,vv in g2_edgesweight.items():  
            newweight=edgeweights1[kk]*g1Norm_EdgeScores[kk]+g2_edgesweight[kk]*g2Norm_EdgeScores[kk] 
           
            if newweight>=txtsim3: 
                g3_newedgeweights[kk]=newweight
        for j in range(features.shape[0]):
            c_temp1=[]
            c_temp1.append(j)
            c_temp1.append(j)
            c_temp1tuple=tuple(c_temp1)
            if str(c_temp1tuple) not in g3_newedgeweights.keys():
                g3_newedgeweights[str(c_temp1tuple)]=1.0  #自连接，0-0，

        
        nxcontents3=[]
        for k3,v3 in g3_newedgeweights.items():
#           
            nxcontents3.append(eval(k3)) 
        Graphg3=nx.DiGraph(nxcontents3) 
        Graph_g3=dgl.from_networkx(Graphg3).to(device)
        Graph_g3.ndata["feat"]= features  
        return Graph_g3,features
def Construct_Graph2Edges(edge_index,textsim,numnodes): 
    edge_index_ = to_undirected(edge_index) 
    deg = degree(edge_index_[1]) 
    degcol = deg[edge_index[1]].to(torch.float32) 
    scol = torch.log(degcol) 
    weights = (scol - scol.min()) / (scol.max() - scol.min())
    print(scol.max(),scol.mean(), scol.min()) 
    wei=weights.tolist()
    count=0
    greatinds=[]
    greatvalues=[]
    for i in range(len(wei)):
#       
        if wei[i]>=textsim:
#          
            count=count+1
            greatinds.append(i)
            greatvalues.append(wei[i])
    
    edge_index= edge_index.tolist() # edge_index_:无向图
    print(len(edge_index[0]),len(edge_index[1])) #1312 1312  2592 2592 注意括号
    new_rows=[]
    new_cols=[]
    for j in range(len(greatinds)): #
        new_rows.append(edge_index[0][greatinds[j]]) #greatinds[5]=6
        new_cols.append(edge_index[1][greatinds[j]])
    print(len(new_cols),len(new_rows)) #1295 1295
    for j in range(numnodes):
        new_rows.append(j) 
        new_cols.append(j)
        greatvalues.append(1.0)
    print(len(new_cols),len(new_rows))  
    nxcontents=[]
    edges_weights=dict()
    for i in range(len(new_rows)):
        templs=[]
        templs.append(new_rows[i]) #对应行
        templs.append(new_cols[i]) #对应列
        temptuple=tuple(templs) 
        nxcontents.append(temptuple)
        edges_weights[str(temptuple)]=greatvalues[i]
    g2_graph=nx.DiGraph(nxcontents) #创建networkx图
    g2=dgl.from_networkx(g2_graph)

    print(len(edges_weights)) 

    return edges_weights, g2  #返回构造好的g2图

if __name__ == "__main__":
#   
    print(f"Training My Dataset with DGL built-in GATConv module.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset='tagmynews'
 
    sys.stdout = Logger(logdir + "{}-MSGCL-D2FSF_testresult.log".format(dataset)) 
    path2 = './data/'
    path = path2+'{}'.format(dataset)  #

    numnodes=7000 #当前节点总数
    out_size = 7#分类的类别数
    current_labels=[0,1,2,3,4,5,6]
    tau=0.4 #GSL loss中的tau值设置 GSL loss1-->(-logxx)
    lamb=0.8
    traintextsim=0.1
    othertextsim=0.1
    learningrate=0.005
    traing2_textsim=0.6  
    train_p=0.6
    other_p=0.55
    WD=0.06

    trainnodes=10*out_size
    valnodes=1000

    featurein_channels=384 
    num_hidden=384
    num_proj_hidden=256 
    activation='prelu'
    base_model='GCNConv' 
    num_layers=2 
    tau=0.4
    k=2
    GCL_weight_decay=1e-5
#     
    print(" GCNConv: ", GCNConv) # GCNConv:  <class 'torch_geometric.nn.conv.gcn_conv.GCNConv'>
    print("feature_inchannels: ",featurein_channels)
    encoder = Encoder(featurein_channels, num_hidden, nn.PReLU(),
                      base_model= GCNConv, k=2).to(device) #k为numer_layers
    print("encoder: ",encoder)
    print(get_activation(activation), GCNConv)

    GCL_model = MYGCL(encoder, num_hidden, num_proj_hidden, tau).to(device)
  
    with open(path+'/traindata/'+'{}_graph_network.pkl'.format(dataset), 'rb') as f:  
        g1=pickle.load(f)
    train_g1=dgl.from_networkx(g1) 
    print("train_g1:",train_g1)

    with open(path+'/traindata/'+'{}_TrainEdgeIndex.pkl'.format(dataset), 'rb') as ff:  
        edge_index=pickle.load(ff)
    with open(path+'/traindata/'+'{}_TrainEdgeIndex_Values.pkl'.format(dataset), 'rb') as ff:  
        edge_index_values=pickle.load(ff)
    print(edge_index,edge_index_values) 
#
    print(edge_index[1][2]) 
  
    
    traing2_edgesweight,train_g2=Construct_Graph2Edges(edge_index, traing2_textsim,trainnodes)
   
    print("train_g2:",train_g2)

    with open(path+'/traindata/'+'{}_trainfeatures.pkl'.format(dataset), 'rb') as f2:  
        feature=pickle.load(f2)

    print(feature,feature.shape) 
    train_g1.ndata["feat"]=feature #batched_graph:第1个图对象
    train_g2.ndata["feat"]=feature
    print("train_g1:",train_g1)
    print("train_g2:",train_g2)
#
   
    with open(path+'/traindata/'+'{}_trainlabels.pkl'.format(dataset), 'rb') as ff:  
        lab=pickle.load(ff)
    print("lab: ",lab) 
    print("lab shape: ",lab.shape)

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

    train_g1.ndata["label"]=lab #batched_graph:第1个图对象 ,train graph对象
    print("train_g1:",train_g1)
    train_g2.ndata["label"]=lab #batched_graph:第1个图对象 ,train graph对象
    print("train_g2:",train_g2)
#！
   
  

    GCL_optimizer = torch.optim.Adam( #配置模型的优化器参数，包括学习率、权重衰减。----届时可以调参
        GCL_model.parameters(),
        lr=learningrate, #第一次调参确定的
                # lr=learning_rate,
        weight_decay=GCL_weight_decay
    )
    with open(path+'/valdata/'+'{}_graph_network.pkl'.format(dataset), 'rb') as ff2:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        valg1=pickle.load(ff2)
    val_g1=dgl.from_networkx(valg1) #batched_graph.ndata['feat'],batched_graph.ndata['label']:
    print("val_g1:",val_g1)

    with open(path+'/valdata/'+'{}_ValEdgeIndex.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        edge_index2=pickle.load(ff)
    with open(path+'/valdata/'+'{}_ValEdgeIndex_Values.pkl'.format(dataset), 'rb') as ff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        edge_index_values2=pickle.load(ff)
    print(edge_index2,edge_index_values2) #g1图中的边和对应的值

    valg2_edgesweight,val_g2=Construct_Graph2Edges(edge_index2,othertextsim,valnodes)
    print("val_g2:", val_g2)

    #加载dataset的features
    with open(path+'/valdata/'+'{}_valfeatures.pkl'.format(dataset), 'rb') as fff2:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        feature2=pickle.load(fff2)
            #     feature=feature.float()
    print(feature2,feature2.shape) #  
           
    val_g1.ndata["feat"]=feature2 #
    val_g2.ndata["feat"]=feature2  
    print("val_g1:",val_g1)
    print("val_g2:",val_g2)
    with open(path+'/valdata/'+'{}_vallabels.pkl'.format(dataset), 'rb') as fff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        lab2=pickle.load(fff)
    print("lab2: ",lab2) 
    valonehotlabs=encode_onehot(lab2)
    print(valonehotlabs,valonehotlabs.shape)
           
    print("lab2 shape: ",lab2.shape) #lab2 shape:  torch.Size([1000])
         
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
    print(feature3,feature3.shape) #  
          
    test_g1.ndata["feat"]=feature3 
    test_g2.ndata["feat"]=feature3 
        
    print("test_g1:",test_g1)
    print("test_g2:",test_g2)
    with open(path+'/testdata/'+'{}_testlabels.pkl'.format(dataset), 'rb') as ffff:  #该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
        lab3=pickle.load(ffff)
    print("lab3: ",lab3) 
    test_g1.ndata["label"]=lab3 #lab3:  tensor([1, 2, 1,  ..., 1, 3, 2])
    test_g2.ndata["label"]=lab3 #
    testonehotlabs=encode_onehot(lab3)
    print(testonehotlabs,testonehotlabs.shape)
        
    print("lab3 shape: ",lab3.shape)
    print(test_g1,test_g2)
    in_size = feature.shape[1] #384

    print("in_size: ",in_size) #in_size:  300
    print("out_size: ",out_size) #out_size:  4

    model =MSGCLD2FSF(in_size, 96, out_size, heads=[4, 4, 6]).to(device) #96*4=384
        
    print("model : ",model)
          
    print("GraphFusion and Training...") 
    train_dataset_list=[]
    val_dataset_list=[]
    train_dataset_list.append(train_g1)
    val_dataset_list.append(val_g1)
    train_dataset_list.append(train_g2) #换成g2图单独测试
    val_dataset_list.append(val_g2)
    print("train_dataset_list:",train_dataset_list)
    train_dataloader = GraphDataLoader(train_dataset_list, batch_size=1) #train图20个，每批2个图--20/2=10批
    val_dataloader = GraphDataLoader(val_dataset_list, batch_size=1)
    print("train_dataloader: ",train_dataloader) 
    print("GraphFusion and Training...")
          
    #只用1个图测试
    test_dataset_list=[]
    test_dataset_list.append(test_g1)
    #同时添加g1和g2图
    test_dataset_list.append(test_g2) #g2图单独test
         
    test_dataloader = GraphDataLoader(test_dataset_list, batch_size=1) #2c个测试图
    #调用train函数，GAT模型开始训练并隔一定epoch，开始val，test
    GraphFusion_Train(train_dataloader, val_dataloader, device, model,train_g1.ndata["feat"]) 

    endtime=time.time() 
    print("cost time:",endtime-starttime)
#     
