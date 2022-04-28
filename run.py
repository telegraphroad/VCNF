import torch
#torch.autograd.set_detect_anomaly(True)
import numpy as np
import normflow as nf

from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import traceback

# Set up model

# Define flows

import argparse
 
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-cb", "--cbase")
parser.add_argument("-mb", "--mbase")
parser.add_argument("-sc", "--scale")
parser.add_argument("-nc", "--ncomp")


args = parser.parse_args()
config = vars(args)
print(config)
cb = float(args.cbase)
mb = float(args.mbase)
sc = float(args.scale)
nc = int(args.ncomp)
print(cb,mb,sc,nc)

# for cb in [1.0001,2.,3.,10.,50.]:
#     for mb in [1.0001,2.,3.,10.,50.]:
#         for sc in [1.,2.,3.,4.,5.]:
#             for nc in [2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,50,100,200,300,500,1000]:        
max_iter = 10
num_samples = 2 * 12
anneal_iter = 10000
annealing = True
show_iter = 1000
# nc = 3
# mb = 1.00015
# cb = 1.00015
# scale = 1.

K = 96
torch.manual_seed(0)

latent_size = 2
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(K):
    s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    flows += [nf.flows.ActNorm(latent_size)]

# Set prior and q0
prior = nf.distributions.target.NealsFunnel()
#q0 = nf.distributions.DiagGaussian(2)
q0 = nf.distributions.base.MultivariateGaussian()


weight = torch.ones(nc,device='cuda')
mbase = torch.tensor(mb,device='cuda')
vbase = torch.tensor(cb,device='cuda')
scale = torch.tensor(sc,device='cuda')

q0 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc)
q1 = deepcopy(q0)                

# Construct flow model
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=prior)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
nfm = nfm.to(device)
nfm = nfm.double()

# Initialize ActNorm
z, _ = nfm.sample(num_samples=2 ** 16)
z_np = z.to('cpu').data.numpy()

# Plot prior distribution
grid_size = 300
xx, yy = torch.meshgrid(torch.linspace(-5, 5, grid_size), torch.linspace(-5, 5, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.double().to(device)
log_prob = prior.log_prob(zz).to('cpu').view(*xx.shape)
prob_prior = torch.exp(log_prob)

# Plot initial posterior distribution
log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0


# Train model


loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-6)
sample0,_ = nfm.sample(90000)
sample0 = pd.DataFrame(sample0.cpu().detach().numpy())


for it in tqdm(range(max_iter)):
    oldm = deepcopy(nfm)
    try:
        optimizer.zero_grad()
        if annealing:
            loss = nfm.reverse_kld(num_samples, beta=np.min([1., 0.001 + it / anneal_iter]))
        else:
            loss = nfm.reverse_alpha_div(num_samples, dreg=True, alpha=1)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

        # Plot learned posterior
        # if (it + 1) % show_iter == 0:
        #     log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
        #     prob = torch.exp(log_prob)
        #     prob[torch.isnan(prob)] = 0

        #     plt.figure(figsize=(15, 15))
        #     plt.pcolormesh(xx, yy, prob.data.numpy())
        #     plt.contour(xx, yy, prob_prior.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        #     plt.gca().set_aspect('equal', 'box')
        #     plt.show()
    except Exception as e:
        print(e)
        traceback.print_exc()                        
        nfm = oldm


# Plot learned posterior distribution
# log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0


sample1 = pd.DataFrame(prior.sample(20000).cpu().detach().numpy())
sample2,_ = nfm.sample(20000)
sample2 = pd.DataFrame(sample2.cpu().detach().numpy())
sample3,_ = q1.forward(20000)
sample3 = pd.DataFrame(sample3.detach().cpu().numpy())
sample4,_ = nfm.q0.forward(20000)
sample4 = pd.DataFrame(sample4.detach().cpu().numpy())

torch.save(nfm, f'/home/samiri/PhD/Synth/VCNF/logs/model_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}.pth')
sample0.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/untrainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}.pth')
sample1.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/target_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}.pth')
sample2.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}.pth')
sample3.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/untrainedbase_{nc}_cb_{cb}_mb_{mb}_scale_{sc}.pth')
sample4.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedbase_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}.pth')
pd.DataFrame(loss_hist).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/losshist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}.pth')
