import torch.nn.functional as F
import torch.nn    as nn
class tabular_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(tabular_net, self).__init__()
        

    


        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)  # AD-NC
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.linear3 = torch.nn.Linear(output_dim, 2)





    def forward(self, x):
        # adjacency1:S  adjacency2:A feature:X

        
        h = F.relu(self.linear1(x))
        h = self.linear2(h)
        h = self.linear3(h)


        logits = h

        return logits