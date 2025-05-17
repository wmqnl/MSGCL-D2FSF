The code and dataset for the paper Multi-source graph contrastive learning with dual-level dynamic fusion of structure and feature for inductive semi-supervised short text classification, implemented in PyTorch.

1）In our codes, The 'data' folder stores the data files we processed. Among them, the meanings of these files are as follows （here, dataset='tagmynews'）:

      *_graph_network.pkl：These files denote the initial training graph, validation graph, or test graph.

       *_TrainEdgeIndex.pkl file和*_TrainEdgeIndex_Values: These files represent the edge indices and weight values of the initial training graph, respectively.

       *_trainfeatures.pkl和*_trainlabels.pkl: These  files denote the short text node embeddings and labels of the initial training graph, respectively.

       *_ValEdgeIndex.pkl file和*_ValEdgeIndex_Values: These files represent the edge indices and weight values of the initial validation graph, respectively.

       *_valfeatures.pkl和*_vallabels.pkl: These files denote the short text node embeddings and labels of the initial validation graph, respectively.

      *_TestEdgeIndex.pkl file和*_TestEdgeIndex_Values: These files represent the edge indices and weight values of the initial test graph, respectively.

      *_testfeatures.pkl和*_testlabels.pkl: These files denote the short text node embeddings and labels of the initial test graph, respectively.

2）Running training and evaluation
      
      python train_MSGCL-D2FSF.py
      
3) For running other datasets:

      Replace the dataset configuration in train_MSGCL-D2FSF.py.

      Adjust the relevant parameters as needed.

Thanks for your attention!