import torch
import torch.nn.functional as F

from torch import nn
from cnn import CNN_Text
from utils import load_emb_matrix
import torch.nn.parameter as parameter
from torch.nn import init
SEED=0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

class Residual(nn.Module):
  def __init__(self, d, fn):
    super(Residual, self).__init__()
    self.fn = fn
    self.projection = nn.Sequential(nn.Linear(d, d), fn, nn.Linear(d, d))

  def forward(self, x):
    return self.fn(x + self.projection(x))


class Net(nn.Module):
  def __init__(self, args):
    super(Net, self).__init__()
    self.word_embed = torch.nn.Embedding(args.n_words, args.word_dim, max_norm=1, padding_idx=0)
    self.word_embed.weight = torch.nn.Parameter(
      torch.from_numpy(load_emb_matrix(args.n_words, args.word_dim, args.data)).float()
    )
   
    self.word_CNN = CNN_Text(args.word_dim, args.n_filters)
 

    self.word_RNN = nn.GRU(num_layers=2,input_size=args.word_dim, hidden_size=50, bidirectional=True, batch_first=True)

    self.tanh1 = nn.Tanh()
    self.w = nn.Parameter(torch.zeros(50* 2))
    self.info_proj = nn.Sequential(nn.Linear(args.n_prop, 100), nn.Tanh())
    self.residual = Residual(200, nn.Tanh())
    self.projection = nn.Linear(300, 100)
  
     
  
  def forward_cnn(self, x):
 
    w_embed = self.word_embed(x)
  
    return self.word_CNN(w_embed)

  def forward_rnn(self, x):

    out_w, _ = self.word_RNN(self.word_embed(x))
    # out_w = torch.mean(out_w, dim=1)
 
    return out_w

  def forward(self, x):
  
    info = x['info']
    info_feature = self.info_proj(info.float())
    word_long= self.forward_cnn(x['desc'])

    word_short = self.forward_rnn(x['short_desc'])
    M = self.tanh1(word_short) 
    alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
    out = word_short * alpha 
    out = torch.sum(out, 1)  


   
    feature = torch.cat([info_feature, word_long], -1)
    feature_res = self.residual(feature)
    feature_res =torch.cat([feature_res, out], -1)
    
    return self.projection(feature_res)






    




     
