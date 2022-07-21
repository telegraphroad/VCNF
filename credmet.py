# Import required packages
#!pip install --upgrade --force-reinstall --no-deps --no-cache-dir git+https://github.com/telegraphroad/VCNF.git
import torch
#torch.autograd.set_detect_anomaly(True)
import numpy as np
import normflow as nf
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import seaborn as sns


from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import numpy
import warnings
from IPython.display import Markdown, display
import pickle
warnings.filterwarnings('ignore')
import pandas as pd
from sdmetrics.single_table import LogisticDetection

from sdmetrics import load_demo
import sdmetrics


import sys,os
sys.path.append(os.getcwd())

import os
import numpy as np 

import torch
import torch.nn as nn

import skimage.io
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from sklearn.metrics import normalized_mutual_info_score
import os
import numpy as np 

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

EPSILON = 1e-10

def marginalPdf(values, bins, sigma):

  residuals = values - bins.unsqueeze(0).unsqueeze(0)
  kernel_values = torch.exp(-0.5*(residuals / sigma).pow(2))
  
  pdf = torch.mean(kernel_values, dim=1)
  normalization = torch.sum(pdf, dim=1).unsqueeze(1) + EPSILON
  pdf = pdf / normalization

  return pdf, kernel_values


def jointPdf(kernel_values1, kernel_values2):

  joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
  normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + EPSILON
  pdf = joint_kernel_values / normalization

  return pdf


def histogram(x, bins, bandwidth):
  '''
    x: tensor of shape BxN
    bins: tensor of length num_bins
    bandwidth: gaussian smoothing factor

    return: normalized histogram of x
  '''
  x = x*255
  pdf, _ = marginalPdf(x.unsqueeze(2), bins, bandwidth)

  return pdf


def histogram2d(x1, x2, bins, bandwidth):
  '''
    values: tensor of shape BxN
    bins: tensor of length num_bins
    bandwidth: gaussian smoothing factor
  '''
  x1 = x1*255
  x2 = x2*255

  pdf1, kernel_values1 = marginalPdf(x1.unsqueeze(2), bins, bandwidth)
  pdf2, kernel_values2 = marginalPdf(x2.unsqueeze(2), bins, bandwidth)

  joint_pdf = jointPdf(kernel_values1, kernel_values2)
  
  return joint_pdf




class MutualInformation(nn.Module):

  def __init__(self, sigma=0.4, num_bins=256, normalize=True):
    super(MutualInformation, self).__init__()

    self.sigma = 2*sigma**2
    self.num_bins = num_bins
    self.normalize = normalize
    self.epsilon = 1e-10

    self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device=device).float(), requires_grad=False)


  def marginalPdf(self, values):

    residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
    
    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
    pdf = pdf / normalization
    
    return pdf, kernel_values


  def jointPdf(self, kernel_values1, kernel_values2):

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
    normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
    pdf = joint_kernel_values / normalization

    return pdf


  def getMutualInformation(self, input1, input2):
    '''
      input1: B, C, H, W
      input2: B, C, H, W

      return: scalar
    '''

    # Torch tensors for images between (0, 1)
    input1 = input1*255
    input2 = input2*255

    B, C, H, W = input1.shape
    assert((input1.shape == input2.shape))

    x1 = input1.view(B, H*W, C)
    x2 = input2.view(B, H*W, C)
    
    pdf_x1, kernel_values1 = self.marginalPdf(x1)
    pdf_x2, kernel_values2 = self.marginalPdf(x2)
    pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

    H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
    H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
    H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

    mutual_information = H_x1 + H_x2 - H_x1x2
    
    if self.normalize:
      mutual_information = 2*mutual_information/(H_x1+H_x2)

    return mutual_information


  def forward(self, input1, input2):
    '''
      input1: B, C, H, W
      input2: B, C, H, W

      return: scalar
    '''
    return self.getMutualInformation(input1, input2)


def qqplot(x, y,ax,color='k',title='', **kwargs):
    x = np.sort(x)
    y = np.sort(y)
    _min = min(x[0], y[0])
    _max = max(x[-1], y[-1])
    
    #ax = plt.gca()
    
    _kwargs = dict(marker='.', alpha=0.5)
    _kwargs.update(kwargs)
    ax.scatter(x, y, **_kwargs)
    ax.plot([_min, _max], [_min, _max], lw=1, linestyle='--',label=title)
    ax.legend()
    return ax

def scatter_plot(xs, ys, zs=None, colors=None, title=None):
    fig = plt.figure(figsize=(9, 8), tight_layout=True)
    canvas = FigureCanvas(fig)

    if zs is None:
        # 2D
        ax = fig.add_subplot(111)
        scatter = ax.scatter(xs, ys, c=colors, cmap='jet')#, vmax=20.0)#, vmin=-11.0, vmax=11.0)
        ax.set_aspect('equal', adjustable='box')
    else:
        # 3D
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=10.0, azim=-80.0)
        ax.set_zlim([-1, 1])
        ax.set_zticks(np.linspace(-1, 1, 5))
        scatter = ax.scatter(xs, ys, zs, c=colors, cmap='jet')#, vmin=-11.0, vmax=11.0)

#     ax.set_xlim([-1, 1])
#     ax.set_xticks(np.linspace(-1, 1, 5))
#     ax.set_ylim([-1, 1])
#     ax.set_yticks(np.linspace(-1, 1, 5))

    if title is not None:
        plt.title(title)

    plt.colorbar(scatter)

    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype='uint8')[..., :3]
    plt.close()

    return image

def get_tail_index(sgd_noise):
    """
    Returns an estimate of the tail-index term of the alpha-stable distribution for the stochastic gradient noise.
    In the paper, the tail-index is denoted by $\alpha$. Simsekli et. al. use the estimator posed by Mohammadi et al. in
    2015.
    :param sgd_noise:
    :return: tail-index term ($\alpha$) for an alpha-stable distribution
    """
    X = sgd_noise.reshape(-1)
    X = X[X.nonzero()]
    K = len(X)
    if len(X.shape)>1:
        X = X.squeeze()
    K1 = int(np.floor(np.sqrt(K)))
    K2 = int(K1)
    X = X[:K1*K2].reshape((K2, K1))
    Y = X.sum(1)
    # X = X.cpu().clone(); Y = Y.cpu().clone()
    a = torch.log(torch.abs(Y)).mean()
    b = (torch.log(torch.abs(X[:int(K2/4),:])).mean()+torch.log(torch.abs(X[int(K2/4):int(K2/2),:])).mean()+torch.log(torch.abs(X[int(K2/2):3*int(K2/4),:])).mean()+torch.log(torch.abs(X[3*int(K2/4):,:])).mean())/4
    alpha_hat = np.log(K1)/(a-b).item()
    return alpha_hat





X = pd.read_csv('/home/samiri/PhD/Synth/VCNF/prep.csv')
X = X.drop(['Unnamed: 0'],1)
c = X.columns
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X)
X.columns = c
X0 = X[X.Class == 0.]
X1 = X[X.Class == 1.]
X0 = X0.drop(['Class'],1)
X1 = X1.drop(['Class'],1)
metrics = sdmetrics.single_table.SingleTableMetric.get_subclasses()
#print(type(t),type(o),t.shape,o.shape,metrics)

# Run all the compatible metrics and get a report

# Original dataset


dir = '/home/samiri/PhD/Synth/VCNF/logs'
for tparam in ['True']:
    #for based in ['GaussianMixture', 'MultivariateGaussian']:
    if True:
        for nc in [100]:                
            for nu in [12,16,32]:
                for sc in [0.,1.]:                          
                    for cb in [3.]:
                        if True:
                            
                            i = -1
                            j = 0
                            for mb in [0.]:
                                #cb = mb
                                prior = nf.distributions.target.NealsFunnel(prop_scale=torch.tensor(20.),
                                prop_shift=torch.tensor(-10.),v1shift = mb, v2shift = 0.)
                                target2 = a = prior.sample(torch.tensor(20000))
                                target = target2.cpu().numpy()
                                clrs = prior.log_prob(target2).exp()
                                j=0
                                i += 1


                                for based in ['MultivariateGaussian', 'GGD','T','GaussianMixture']:
                                    #print(i,j)
                                    try:
                                        if based in ['GaussianMixture', 'MultivariateGaussian']:
                                            cb = mb
                                            nc = 100
                                        if based in ['GGD', 'T']:
                                            nc = 1
                                            cb = 3.0
                                        
                                        pref = f'nc_{nc}_mb_{mb}_cb_{cb}_{tparam}_nunit_{nu}_{based}'
                                        #print(i,j, pref)
                                        print(pref)
                                        # model = torch.load(f'{dir}/model_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
                                        # untmodel = pd.read_csv(f'{dir}/untrainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth').drop(['Unnamed: 0'],1).values
                                        # #target = pd.read_csv(f'/home/samiri/PhD/Synth/VCNF/logs/target_nc_{nc}_cb_{mb}_mb_{mb}_scale_{sc}.pth')[['0','1']].values
                                        tmodel = pd.DataFrame(pd.read_csv(f'{dir}/trainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth').drop(['Unnamed: 0'],1).values)
                                        tmodel.columns = X0.columns
                                        if sc == 0.:
                                            target = X0
                                            print(0)
                                            target = target.sample(len(tmodel))
                                            
                                            fl = tmodel
                                        else:
                                            target = X1
                                            fl=tmodel.sample(len(target))
                                            print(1)
                                        #fl = .sample(len(target))
                                        # untbase = pd.read_csv(f'{dir}/untrainedbase_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')[['0','1']].values
                                        # tbase = pd.read_csv(f'{dir}/trainedbase_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')[['0','1']].values
                                        # losshist = pd.read_csv(f'{dir}/losshist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth').drop(['Unnamed: 0'],1).values
                                        # gzarr = pickle.load(open(f'{dir}/z_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth', "rb" ))#.drop(['Unnamed: 0'],1).values
                                        # gzparr = pickle.load(open(f'{dir}/zp_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth', "rb" ))#.drop(['Unnamed: 0'],1).values
                                        # phist = pd.read_csv(f'{dir}/phist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth').drop(['Unnamed: 0'],1).values
                                        # grads = torch.load(f'{dir}/grads_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
                                        # wb = torch.load(f'{dir}/wb_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
                                        # gmeans = grads
                                        # for g in grads:
                                        #     gmeans.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in g.values() if i is not None]) if i != 0]))
                                        #indx = np.random.choice(untmodel.shape[0], 20000, replace=False)  
                                        #untmodel = untmodel[indx,:]
                                        # ttemp = pd.DataFrame(tmodel)
                                        # tmodel = ttemp[(ttemp[1]<11) & (ttemp[1]>-11)].values                                
                                        # tt = get_tail_index(torch.tensor(target))
                                        # bt = get_tail_index(torch.tensor(untbase))
                                        # mtt = get_tail_index(torch.tensor(tmodel))
                                        # mit = get_tail_index(torch.tensor(untmodel))
                                        # tails.append([based,nc,nu,tparam,cb,mb,sc,tt-bt,tt-mit,tt-mtt,tt])
                                        print(fl.shape,target.shape)
                                        m = sdmetrics.compute_metrics(metrics, target, fl)
                                        m = m[m.raw_score == m.raw_score]
                                        m.to_csv(f'{dir}/metrics_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
                                        print(m)

                                    except Exception as e:
                                        1
                                        print(e)
                                        #plt.close(fig)

