import torch
import torch.nn as nn
import torch.nn.functional as F

#model
class QNetWork(nn.Module):
    """
    transform a state to action value
    """
    def __init__(self,state_size,action_size,hidden_layers=[256,128]):
        """
        state_size: the size of state space
        action_size: the dimenson
        hidden_layers: a list of nodes
        """
        super(QNetWork,self).__init__()
        node_list = [state_size] + hidden_layers + [action_size]
        self.layers = nn.ModuleList([nn.Linear(node_list[i],node_list[i+1]) for i in range(len(node_list)-1)])
    
    def forward(self,x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x