import torch
#torch.autograd.set_detect_anomaly(True)
import numpy as np
import normflow as nf
import copy
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import traceback
import pickle

# Set up model

# Define flows

import argparse
 
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-cb", "--cbase")
parser.add_argument("-mb", "--mbase")
parser.add_argument("-sc", "--scale")
parser.add_argument("-nc", "--ncomp")
parser.add_argument("-nu", "--nunit")
parser.add_argument("-b", "--base")
parser.add_argument("-trainable", "--trainablebase")

args = parser.parse_args()
config = vars(args)
print(config)
cb = float(args.cbase)
mb = float(args.mbase)
sc = float(args.scale)
nc = int(args.ncomp)
nu = int(args.nunit)
tparam = bool(int(args.trainablebase))
based = str(args.base)
print(cb,mb,sc,nc,nu,tparam,based)

# for cb in [1.0001,2.,3.,10.,50.]:
#     for mb in [1.0001,2.,3.,10.,50.]:
#         for sc in [1.,2.,3.,4.,5.]:
#             for nc in [2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,50,100,200,300,500,1000]:        
max_iter = 30000
num_samples = 2 ** 10
anneal_iter = 14000
annealing = True
show_iter = 20
# nc = 3
# mb = 1.00015
# cb = 1.00015
# scale = 1.

K = nu
torch.manual_seed(0)

latent_size = 2
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(K):
    s = nf.nets.MLP([latent_size, 8 * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 8 * latent_size, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    flows += [nf.flows.ActNorm(latent_size)]

# Set prior and q0
prior = nf.distributions.target.GMixture(dim=2)
#q0 = nf.distributions.DiagGaussian(2)
q0 = nf.distributions.base.MultivariateGaussian()


weight = torch.ones(nc,device='cuda')
mbase = torch.tensor(mb,device='cuda')
vbase = torch.tensor(cb,device='cuda')
scale = torch.tensor(sc,device='cuda')

#q0 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,trainable = tparam)
print('~~~~~~~~',based)
if based == 'GaussianMixture':
    q0 = nf.distributions.base.GaussianMixture(n_modes = nc, dim = 2, trainable=tparam)
    q1 = nf.distributions.base.GaussianMixture(n_modes = nc, dim = 2, trainable=tparam)
elif based == 'GMM':
    q0 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,trainable = tparam)
    q1 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,trainable = tparam)
elif based == 'T':
    q0 = nf.distributions.base.T(n_dim=2, df = cb,trainable = tparam)
    q1 = nf.distributions.base.T(n_dim=2, df = cb,trainable = tparam)
elif based == 'GGD':
    q0 = nf.distributions.base.GGD(n_dim=2, beta = cb,trainable = tparam)
    q1 = nf.distributions.base.GGD(n_dim=2, beta = cb,trainable = tparam)

elif based == 'MultivariateGaussian':
    q0 = nf.distributions.base.MultivariateGaussian(trainable=tparam)
    q1 = nf.distributions.base.MultivariateGaussian(trainable=tparam)

with torch.no_grad():
    sample3,_ = q1.forward(20000)
    #print(sample3.shape)
    sample3 = pd.DataFrame(sample3.detach().cpu().numpy())

# Construct flow model
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=prior)
nfmBest = nf.NormalizingFlow(q0=q0, flows=flows, p=prior)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
nfm = nfm.to(device)
nfmBest = nfmBest.to(device)
nfmBest = nfmBest.double()

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
if tparam:
    optimizer2 = torch.optim.Adam(q0.parameters(), lr=1e-4, weight_decay=1e-6)

sample0,_ = nfm.sample(20000)
sample0 = pd.DataFrame(sample0.cpu().detach().numpy())
gzarr = []
gzparr = []
phist = []
phistg = []
grads = []
gradssteps = []
wb = []
logp = []
logq = []
tgrads = []
closs = 1e20
nfmBest.state_dict = nfm.state_dict
oldm = nfm.state_dict()
oldp = q0.parameters

for it in tqdm(range(max_iter)):
    try:
        nfm.sample(2)
        optimizer.zero_grad()
        if annealing:
            loss,zarr,zparr,logpep,logqep = nfm.reverse_kld(num_samples, beta=np.min([1., 0.001 + it / anneal_iter]), extended = True)
        else:
            loss = nfm.reverse_alpha_div(num_samples, dreg=True, alpha=1)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            
            loss.backward(retain_graph=True)
            #grads.append({n:p.grad for n, p in nfm.named_parameters()})

            with torch.no_grad():
                a = [p.grad for n, p in nfm.named_parameters()]
                asteps = []
                for lctr in range(0,nu):
                    asteps.append([p.grad for n, p in nfm.named_parameters() if (('flows.'+str(lctr)+'.' in n) or ('flows.'+str(lctr+1)+'.' in n))])
                agrads = []
                #print('pgrad',q0.p.grad,'locgrad',q0.loc.grad,'scalegrad',q0.scale.grad)
                #if str(q0) == 'GGD()':
                for lg in asteps:
                    agrads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in lg if i is not None]) if i != 0]))
                #    tgrads.append(q0.p.grad,q0.loc.grad,q0.scale.grad)
                #print([[n,p] for n, p in nfm.named_parameters()])
                #print(a[3].mean(),a[4].mean())
                grads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in a if i is not None]) if i != 0]))
                gradssteps.append(agrads)

            optimizer.step()
            if tparam:
                optimizer2.step()

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        #print(logpep.median().detach().cpu().numpy())
        logp.append(logpep.detach().cpu().numpy())
        logq.append(logqep.detach().cpu().numpy())
        
        pm = {n:p.detach().cpu().numpy() for n, p in nfm.named_parameters()}


        #
        
        #print(nfm.q0.mbase.detach().cpu().item())
        phistg.append([a.grad.detach().cpu().numpy() for a in q0.parameters() if a.grad is not None])

        # Plot learned posterior
        if (it + 1) % show_iter == 0:
            wb.append(pm)
            
            


            gzarr.append(zarr)
            gzparr.append(zparr)
            phist.append([a.detach().cpu().numpy() for a in q0.parameters()])
            # if based == 'GaussianMixture':
            #     phistg.append([nfm.q0.loc.grad,nfm.q0.log_scale.grad,nfm.q0.weight_scores.grad])

            # elif based == 'GMM':
            #     phistg.append([nfm.q0.mbase.grad,nfm.q0.vbase.grad,nfm.q0.scale.grad,nfm.q0.weight.grad])
            # elif based == 'GGD':
            #     phistg.append([nfm.q0.loc.grad,nfm.q0.scale.grad,nfm.q0.p.grad])
            # elif based == 'T':
            #     phistg.append([nfm.q0.df.grad])

            # elif based == 'MultivariateGaussian':
            #     phistg.append([nfm.q0.loc.grad,nfm.q0.scale.grad])

            # if based == 'GaussianMixture':
            #     phist.append([nfm.q0.loc,nfm.q0.log_scale,nfm.q0.weight_scores])

            # elif based == 'GMM':
            #     phist.append([nfm.q0.mbase,nfm.q0.vbase,nfm.q0.scale,nfm.q0.weight])
            # elif based == 'GGD':
            #     phist.append([nfm.q0.loc,nfm.q0.scale,nfm.q0.p])
            # elif based == 'T':
            #     phist.append([nfm.q0.df])

            # elif based == 'MultivariateGaussian':
            #     phist.append([nfm.q0.loc.grad,nfm.q0.scale.grad])
        #     log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
        #     prob = torch.exp(log_prob)
        #     prob[torch.isnan(prob)] = 0

        #     plt.figure(figsize=(15, 15))
        #     plt.pcolormesh(xx, yy, prob.data.numpy())
        #     plt.contour(xx, yy, prob_prior.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        #     plt.contour(xx, yy, prob_prior.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        #     plt.gca().set_aspect('equal', 'box')
        #     plt.show()
        if loss.to('cpu').data.item()<closs:
            try:
                nfm.sample(2)
                closs = np.abs(loss.to('cpu').data.item())
                nfmBest.load_state_dict(copy.deepcopy(nfm.state_dict()))
                q1.parameters = q0.parameters
            except Exception as e:
                print('==============loss down but sampling failed!')
                print(e)
                
            #print('BU')
            try:
                nfm.sample(2)
                oldm = copy.deepcopy(nfm.state_dict())
                oldp = q0.parameters
            except Exception as e:
                print('==============loss down but sampling failed!')
                print(e)

        
    except Exception as e:
        print('==============training step failed!')
        print(e)
        traceback.print_exc()                        
        nfm.load_state_dict(oldm)
        q0.parameters = oldp


# Plot learned posterior distribution
# log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0


# sample1 = pd.DataFrame(prior.sample(20000).cpu().detach().numpy())
# sample2,_ = nfm.sample(20000)
# sample2 = pd.DataFrame(sample2.cpu().detach().numpy())
# sample4,_ = nfm.q0.forward(20000)
# sample4 = pd.DataFrame(sample4.detach().cpu().numpy())
try:
    sample1 = pd.DataFrame(prior.sample(20000).cpu().detach().numpy())
    sample2,_ = nfmBest.sample(20000)
    sample2 = pd.DataFrame(sample2.cpu().detach().numpy())
    sample20,_ = nfm.sample(20000)
    sample20 = pd.DataFrame(sample20.cpu().detach().numpy())
    sample4,_ = nfmBest.q0.forward(20000)
    sample4 = pd.DataFrame(sample4.detach().cpu().numpy())

    torch.save(nfmBest, f'/home/samiri/PhD/Synth/VCNF/logs/model_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    sample0.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/untrainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    sample1.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/target_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    sample2.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedmodelB_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    sample20.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    sample3.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/untrainedbase_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    sample4.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedbase_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    pd.DataFrame(loss_hist).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/losshist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    #pickle.dump(gzarr, open( f'/home/samiri/PhD/Synth/VCNF/logs/z_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth', 'wb'))
    #pickle.dump(gzparr, open( f'/home/samiri/PhD/Synth/VCNF/logs/zp_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth', 'wb'))
    #pd.DataFrame(gzarr).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/z_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    pd.DataFrame(logp).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/logp_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    pd.DataFrame(logq).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/logq_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    #pd.DataFrame(gzparr).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/zp_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    torch.save(phist,f'/home/samiri/PhD/Synth/VCNF/logs/phist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    torch.save(phistg,f'/home/samiri/PhD/Synth/VCNF/logs/phistg_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    torch.save(grads, f'/home/samiri/PhD/Synth/VCNF/logs/grads_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    torch.save(gradssteps, f'/home/samiri/PhD/Synth/VCNF/logs/gradssteps_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    torch.save(wb, f'/home/samiri/PhD/Synth/VCNF/logs/wb_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
    #torch.save(tgrads, f'/home/samiri/PhD/Synth/VCNF/logs/tgrads_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
except Exception as e:
    print(e)
    traceback.print_exc()                        
