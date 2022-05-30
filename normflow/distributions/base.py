import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
from torch.distributions import ExponentialFamily,Categorical,constraints,MultivariateNormal, Independent
from torch.distributions.utils import _standard_normal,broadcast_all
from numbers import Real, Number
import math
import copy
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from typing import Dict
from scipy.stats import gennorm

from .. import flows

class GenNormal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.
    Example::
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])
    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'p': constraints.real}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def exponent(self):
        return self.p

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale,p, validate_args=None):
        self.loc, self.scale, self.p = broadcast_all(loc, scale, p)
        #print('11111',p)
        
        self.p = p
        self.loc = loc
        self.scale = scale
        #print('2222',self.p)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(GenNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GenNormal, _instance)

        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(self.loc.shape)
        new.scale = self.scale.expand(self.scale.shape)
        new.p = self.p.expand(self.p.shape)
        super(GenNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    def rsample(self, sample_shape=torch.Size()):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print('~~~~~~shape1',shape)
        shape = self._extended_shape(sample_shape)
        #print('~~~~~~shape2',shape)
        ipower = 1.0 / self.p
        
        ipower = ipower.mean()#.cpu()
        gamma_dist = torch.distributions.Gamma(ipower, 1.0)
        
        gamma_sample = gamma_dist.rsample(shape)#.cpu()
        
        binary_sample = torch.randint(low=0, high=2, size=shape, dtype=self.loc.dtype) * 2 - 1
        
        #print('~~~~~~',binary_sample.shape,gamma_sample.shape)
              
        if len(binary_sample.shape) ==  len(gamma_sample.shape) - 1:
            gamma_sample = gamma_sample.squeeze(len(gamma_sample.shape) - 1)
            
        #print('~~~',binary_sample.shape,gamma_sample.shape)
        sampled = binary_sample.to(device) * torch.pow(torch.abs(gamma_sample).to(device), ipower)
        
        #print(self.loc.detach().cpu().numpy(),':::::',self.scale.detach().cpu().numpy(),':::::',self.p.detach().cpu().numpy())
        #return self.loc.to(device) + self.scale.to(device) * sampled.to(device)
        return self.loc + self.scale * sampled

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        #print('shape',shape)
        with torch.no_grad():
            ipower = 1.0 / self.p
            ipower = ipower#.cpu()
            gamma_dist = torch.distributions.Gamma(ipower, 1.0)
            gamma_sample = gamma_dist.sample(shape)#.cpu()
            binary_sample = (torch.randint(low=0, high=2, size=shape, dtype=self.loc.dtype) * 2 - 1)
            if (len(gamma_sample.shape) == len(binary_sample.shape) + 1) and gamma_sample.shape[-1]==gamma_sample.shape[-2]:
              gamma_sample = gamma_dist.sample(shape[0:-1])#.cpu()
              
#             
#             
#             
#             
            if type(ipower) == torch.Tensor:
              sampled = binary_sample.to(gamma_sample.device).squeeze() * torch.pow(torch.abs(gamma_sample.squeeze()).to(gamma_sample.device), ipower.to(gamma_sample.device))
            else:
              sampled = binary_sample.squeeze() * torch.pow(torch.abs(gamma_sample.squeeze()), torch.FloatTensor(ipower))
            #print(self.loc.item(),':::::',self.scale.item(),':::::',self.p.item())
            return self.loc + self.scale * sampled

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        return (-((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi)))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)
                                    
                                    

class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """
    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """
        Samples from base distribution and calculates log probability
        :param num_samples: Number of samples to draw from the distriubtion
        :return: Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """
        Calculate log probability of batch of samples
        :param z: Batch of random variables to determine log probability for
        :return: log probability for each batch element
        """
        raise NotImplementedError


class GMM(nn.Module):
    
    def __init__(self, weights, mbase,vbase, scale, n_cell=8, shift=0, dim=2, trainable=False):
        super(GMM, self).__init__()
        self.weight = nn.Parameter(weights, requires_grad = True)
        self.mbase = nn.Parameter(mbase, requires_grad = True)
        self.vbase = nn.Parameter(vbase, requires_grad = True)
        self.scale = nn.Parameter(scale, requires_grad = False)
        self.grid = torch.arange(1, n_cell+1,device='cuda')
        self.shift = shift
        self.n_cell = n_cell
        self.dim = dim
        with torch.no_grad():
            means = self.trsf_gridm()
            std = self.trsf_gridv()
            mix = D.Categorical(self.weight)
            comp = D.Independent(D.Normal(means, std+0.001), 1)
            self.gmm = D.MixtureSameFamily(mix, comp)

    
    def trsf_gridm(self):
        trsf = (
            torch.log(self.scale * self.grid + self.shift) 
            / torch.log(self.mbase)
            ).reshape(-1, 1)

        return trsf.expand(self.n_cell, self.dim)

    def trsf_gridv(self):
        trsf = (
            torch.log(self.scale * self.grid + self.shift) 
            / torch.log(self.vbase)
            ).reshape(-1, 1)
        return trsf.expand(self.n_cell, self.dim)
    
    def forward(self, num_samples=1):
        print('8888888888')
        means = self.trsf_gridm() 
        std = self.trsf_gridv()
        mix = D.Categorical(self.weight)
        comp = D.Independent(D.Normal(means, std+0.001), 1)
        self.gmm = D.MixtureSameFamily(mix, comp)
        with torch.no_grad():
            samples = self.gmm.sample([num_samples]).double()
        return samples.T, self.gmm.log_prob(samples)

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)
        return self.gmm.log_prob(z)


# class GMM(nn.Module):
    
#     def __init__(self, weights, mbase,vbase, scale, n_cell=8, shift=0, dim=2, trainable=False):
#         super(GMM, self).__init__()
#         self.weight = nn.Parameter(weights, requires_grad = False)
#         self.mbase = nn.Parameter(mbase, requires_grad = trainable)
#         self.vbase = nn.Parameter(vbase, requires_grad = False)
#         self.scale = nn.Parameter(scale, requires_grad = False)
#         self.grid = torch.arange(1, (n_cell+1)/2,device='cuda')
#         self.shift = shift
#         self.n_cell = n_cell
#         self.dim = dim
#         with torch.no_grad():
#             means = self.trsf_gridm()
#             std = self.trsf_gridv()
#             mix = D.Categorical(self.weight)
#             comp = D.Independent(D.Normal(means, std+0.001), 1)
#             self.gmm = D.MixtureSameFamily(mix, comp)

    
#     def trsf_gridm(self):
#         trsf = torch.pow(self.mbase, self.grid)
#         trsf = self.scale * (trsf - trsf.min())/(trsf.max()-trsf.min())
#         trsf = torch.concat([-trsf,trsf]).sort()[0].reshape(-1,1)
#         return trsf.expand(self.n_cell, self.dim)

#     def trsf_gridv(self):
#         trsf = torch.pow(self.vbase, self.grid)
#         trsf = self.scale * (trsf - trsf.min())/(trsf.max()-trsf.min())
#         trsf = torch.concat([torch.flip(trsf,[0]),trsf]).abs().reshape(-1,1)
#         return trsf.expand(self.n_cell, self.dim)
    
#     def forward(self, num_samples=1):

#         means = self.trsf_gridm() 
#         std = self.trsf_gridv()
#         mix = D.Categorical(self.weight)
#         comp = D.Independent(D.Normal(means, std+0.001), 1)
#         self.gmm = D.MixtureSameFamily(mix, comp)
#         with torch.no_grad():
#             samples = self.gmm.sample([num_samples]).double()
#         return samples, self.gmm.log_prob(samples)

#     def log_prob(self, z):
#         #print('~~~0',self.mbase)
#         return self.gmm.log_prob(z)



class MultivariateGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, n_components = 3, trainable=False, loc = 0., scale = 1.0):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        self.n_components = n_components
        self.n_dim = n_dim

        with torch.no_grad():
            if trainable:
                
                self.loc = nn.Parameter(torch.zeros(self.n_dim,dtype=torch.double,device='cuda') + loc, requires_grad = True)
                self.scale = nn.Parameter(torch.eye(self.n_dim,dtype=torch.double,device='cuda') + scale, requires_grad = True)
            else:
                
                self.register_buffer("loc", torch.zeros(self.n_dim,dtype=torch.double,device='cuda'))
                self.register_buffer("scale", torch.eye(self.n_dim,dtype=torch.double,device='cuda'))

        self.mvn = D.MultivariateNormal(self.loc, self.scale)
        
    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.mvn.sample([num_samples])
        #print(z)
        log_prob= self.mvn.log_prob(z)
        return z, log_prob

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)

        return self.mvn.log_prob(z)


class MultivariateMixtureofGaussians(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, n_components = 3, trainable=False, prop_scale=torch.tensor(6.),
                 prop_shift=torch.tensor(-3.)):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        self.n_components = n_components
        self.n_dim = n_dim
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)
        self.max_log_prob = 0.

        self.gmm = []
        with torch.no_grad():
            for _ in range(self.n_dim):
                if trainable:
                    self.w = nn.Parameter(torch.ones(self.n_components,dtype=torch.double,device='cuda'), requires_grad = True)
                    self.loc = nn.Parameter(torch.zeros(self.n_components,dtype=torch.double,device='cuda'), requires_grad = True)
                    self.scale = nn.Parameter(torch.ones(self.n_components,dtype=torch.double,device='cuda'), requires_grad = True)
                    mix = D.Categorical(self.w)
                    comp = D.Normal(self.loc, self.scale)
                    self.gmm.append(D.MixtureSameFamily(mix, comp))

                else:
                    self.register_buffer("w", torch.ones(self.n_components,dtype=torch.double,device='cuda'))
                    self.register_buffer("loc", torch.zeros(self.n_components,dtype=torch.double,device='cuda'))
                    self.register_buffer("scale", torch.ones(self.n_components,dtype=torch.double,device='cuda'))
                    mix = D.Categorical(self.w)
                    comp = D.Normal(self.loc, self.scale)
                    self.gmm.append(D.MixtureSameFamily(mix, comp))


    def rejection_sampling(self, num_steps=1):
        """
        Perform rejection sampling on image distribution
        :param num_steps: Number of rejection sampling steps to perform
        :return: Accepted samples
        """
        eps = torch.rand((num_steps, 1), dtype=self.prop_scale.dtype,
                         device=self.prop_scale.device).squeeze()
        z_ = self.prop_scale * eps + self.prop_shift
        prob = torch.rand(num_steps, dtype=self.prop_scale.dtype,
                          device=self.prop_scale.device)
        #print('++++',z_.shape,prob.shape,len(self.log_prob(z_.squeeze())))
        prob_ = torch.exp(self.log_prob(z_.squeeze()) - self.max_log_prob)
        accept = prob_ > prob
        #print('++++++',accept.shape,z_.shape)
        z = z_[accept]
        return z

    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        z = torch.zeros((0, 1), dtype=self.prop_scale.dtype,
                        device=self.prop_scale.device).squeeze()
        
        while len(z) < num_samples:
            
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind]], 0)
        return z



    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.sample(num_samples)
        #print(z)
        log_prob= self.log_prob(z)
        return z, log_prob

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)
        lp = 0.
        for g in self.gmm:
            lp += g.log_prob(z)
        #print('~~~~',lp.shape,g.log_prob(z).shape)
        return lp

    def log_prob_gmm(self, z):

        return [i.log_prob(z) for i in self.gmm]




class MixtureofMultivariateGaussians(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, n_components = 3, trainable=False):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        self.n_components = n_components
        self.n_dim = n_dim

        with torch.no_grad():
            if trainable:
                self.w = nn.Parameter(torch.ones((self.n_components,),dtype=torch.double,device='cuda'), requires_grad = True)
                self.loc = nn.Parameter(torch.zeros((self.n_components,self.n_dim),dtype=torch.double,device='cuda'), requires_grad = True)
                self.scale = nn.Parameter(torch.ones((self.n_components,self.n_dim),dtype=torch.double,device='cuda'), requires_grad = True)
            else:
                self.register_buffer("w", torch.ones((self.n_components,),dtype=torch.double,device='cuda'))
                self.register_buffer("loc", torch.zeros((self.n_components,self.n_dim),dtype=torch.double,device='cuda'))
                self.register_buffer("scale", torch.ones((self.n_components,self.n_dim),dtype=torch.double,device='cuda'))

        mix = D.Categorical(self.w)
        comp = D.Independent(D.Normal(self.loc, self.scale), 1)
        self.gmm = D.MixtureSameFamily(mix, comp)#univ
        #print('~~~1',self.gmm.mixture_distribution.probs)


    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.gmm.sample([num_samples])
        #print(z)
        log_prob= self.gmm.log_prob(z)
        return z, log_prob

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)

        return self.gmm.log_prob(z)


class GGD(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, beta=2., trainable=False):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        
        self.n_dim = n_dim
        self.beta = beta
        with torch.no_grad():
            if trainable:
                self.p = nn.Parameter(torch.ones((self.n_dim,),dtype=torch.double,device='cuda') + self.beta, requires_grad = True)
                self.loc = nn.Parameter(torch.zeros((self.n_dim),dtype=torch.double,device='cuda'), requires_grad = True)
                self.scale = nn.Parameter(torch.ones((self.n_dim),dtype=torch.double,device='cuda'), requires_grad = True)
            else:
                self.register_buffer("p", torch.ones((self.n_dim,),dtype=torch.double,device='cuda'))
                self.register_buffer("loc", torch.zeros((self.n_dim),dtype=torch.double,device='cuda'))
                self.register_buffer("scale", torch.ones((self.n_dim),dtype=torch.double,device='cuda'))

        
        #print('*************',self.n_dim,self.loc,self.scale,self.p)
        self.ggd = GenNormal(self.loc,self.scale,self.p)#univ
        #print('~~~1',self.gmm.mixture_distribution.probs)


    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.ggd.sample([num_samples])
        #print(z)
        log_prob= self.ggd.log_prob(z).mean(axis=1)
        return z, log_prob

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)

        return self.ggd.log_prob(z).mean(axis=1)


class T(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, df = 2., trainable=False):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        
        self.n_dim = n_dim
        self.df = df


        with torch.no_grad():
            if trainable:
                self.df = nn.Parameter(torch.zeros((self.n_dim,),dtype=torch.double,device='cuda') + self.df, requires_grad = True)
            else:
                self.register_buffer("df", torch.ones((self.n_dim,) + self.df,dtype=torch.double,device='cuda'))

        self.t = D.StudentT(self.df)#univ
        #print('~~~1',self.gmm.mixture_distribution.probs)


    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.t.sample([num_samples])
        #print(z)
        log_prob= self.t.log_prob(z).mean(axis=1)
        return z, log_prob

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)

        return self.t.log_prob(z).mean(axis=1)

        
        
class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, shape, trainable=True):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1):
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype,
                          device=self.loc.device)
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - torch.sum(log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1)))
        return z, log_p

    def log_prob(self, z):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = - 0.5 * self.d * np.log(2 * np.pi)\
                - torch.sum(log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
                            list(range(1, self.n_dim + 1)))
        return log_p


class UniformGaussian(BaseDistribution):
    """
    Distribution of a 1D random variable with some entries having a uniform and
    others a Gaussian distribution
    """
    def __init__(self, ndim, ind, scale=None):
        """
        Constructor
        :param ndim: Int, number of dimensions
        :param ind: Iterable, indices of uniformly distributed entries
        :param scale: Iterable, standard deviation of Gaussian or width of
        uniform distribution
        """
        super().__init__()
        self.ndim = ndim

        # Set up indices and permutations
        self.ndim = ndim
        if torch.is_tensor(ind):
            self.register_buffer('ind', torch._cast_Long(ind))
        else:
            self.register_buffer('ind', torch.tensor(ind, dtype=torch.long))

        ind_ = []
        for i in range(self.ndim):
            if not i in self.ind:
                ind_ += [i]
        self.register_buffer('ind_', torch.tensor(ind_, dtype=torch.long))

        perm_ = torch.cat((self.ind, self.ind_))
        inv_perm_ = torch.zeros_like(perm_)
        for i in range(self.ndim):
            inv_perm_[perm_[i]] = i
        self.register_buffer('inv_perm', inv_perm_)

        if scale is None:
            self.register_buffer('scale', torch.ones(self.ndim))
        else:
            self.register_buffer('scale', scale)

    def forward(self, num_samples=1):
        z = self.sample(num_samples)
        return z, self.log_prob(z)

    def sample(self, num_samples=1):
        eps_u = torch.rand((num_samples, len(self.ind)), dtype=self.scale.dtype,
                           device=self.scale.device) - 0.5
        eps_g = torch.randn((num_samples, len(self.ind_)), dtype=self.scale.dtype,
                            device=self.scale.device)
        z = torch.cat((eps_u, eps_g), -1)
        z = z[..., self.inv_perm]
        return self.scale * z

    def log_prob(self, z):
        log_p_u = torch.broadcast_to(-torch.log(self.scale[self.ind]), (len(z), -1))
        log_p_g = - 0.5 * np.log(2 * np.pi) - torch.log(self.scale[self.ind_]) \
                  - 0.5 * torch.pow(z[..., self.ind_] / self.scale[self.ind_], 2)
        return torch.sum(log_p_u, -1) + torch.sum(log_p_g, -1)


class ClassCondDiagGaussian(BaseDistribution):
    """
    Class conditional multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, shape, num_classes):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        :param num_classes: Number of classes to condition on
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.perm = [self.n_dim] + list(range(self.n_dim))
        self.d = np.prod(shape)
        self.num_classes = num_classes
        self.loc = nn.Parameter(torch.zeros(*self.shape, num_classes))
        self.log_scale = nn.Parameter(torch.zeros(*self.shape, num_classes))
        self.temperature = None # Temperature parameter for annealed sampling

    def forward(self, num_samples=1, y=None):
        if y is not None:
            num_samples = len(y)
        else:
            y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
        if y.dim() == 1:
            y_onehot = torch.zeros((self.num_classes, num_samples), dtype=self.loc.dtype,
                                   device=self.loc.device)
            y_onehot.scatter_(0, y[None], 1)
            y = y_onehot
        else:
            y = y.t()
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype,
                          device=self.loc.device)
        loc = (self.loc @ y).permute(*self.perm)
        log_scale = (self.log_scale @ y).permute(*self.perm)
        if self.temperature is not None:
            log_scale = np.log(self.temperature) + log_scale
        z = loc + torch.exp(log_scale) * eps
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - torch.sum(log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1)))
        return z, log_p

    def log_prob(self, z, y):
        if y.dim() == 1:
            y_onehot = torch.zeros((self.num_classes, len(y)), dtype=self.loc.dtype,
                                   device=self.loc.device)
            y_onehot.scatter_(0, y[None], 1)
            y = y_onehot
        else:
            y = y.t()
        loc = (self.loc @ y).permute(*self.perm)
        log_scale = (self.log_scale @ y).permute(*self.perm)
        if self.temperature is not None:
            log_scale = np.log(self.temperature) + log_scale
        log_p = - 0.5 * self.d * np.log(2 * np.pi)\
                - torch.sum(log_scale + 0.5 * torch.pow((z - loc) / torch.exp(log_scale), 2),
                            list(range(1, self.n_dim + 1)))
        return log_p


class GlowBase(BaseDistribution):
    """
    Base distribution of the Glow model, i.e. Diagonal Gaussian with one mean and
    log scale for each channel
    """
    def __init__(self, shape, num_classes=None, logscale_factor=3.):
        """
        Constructor
        :param shape: Shape of the variables
        :param num_classes: Number of classes if the base is class conditional,
        None otherwise
        :param logscale_factor: Scaling factor for mean and log variance
        """
        super().__init__()
        # Save shape and related statistics
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.num_pix = np.prod(shape[1:])
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        self.logscale_factor = logscale_factor
        # Set up parameters
        self.loc = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        self.loc_logs = nn.Parameter(torch.zeros(1, self.shape[0],
                                                 *((self.n_dim - 1) * [1])))
        self.log_scale = nn.Parameter(torch.zeros(1, self.shape[0],
                                                  *((self.n_dim - 1) * [1])))
        self.log_scale_logs = nn.Parameter(torch.zeros(1, self.shape[0],
                                                       *((self.n_dim - 1) * [1])))
        # Class conditional parameter if needed
        if self.class_cond:
            self.loc_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))
            self.log_scale_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))
        # Temperature parameter for annealed sampling
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        # Prepare parameter
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(self.log_scale_logs * self.logscale_factor)
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.loc.dtype,
                                       device=self.loc.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(y.size(0), self.shape[0],
                                               *((self.n_dim - 1) * [1]))
            log_scale = log_scale + (y @ self.log_scale_cc).view(y.size(0), self.shape[0],
                                                                 *((self.n_dim - 1) * [1]))
        if self.temperature is not None:
            log_scale = log_scale + np.log(self.temperature)
        # Sample
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype,
                          device=self.loc.device)
        z = loc + torch.exp(log_scale) * eps
        # Get log prob
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - self.num_pix * torch.sum(log_scale, dim=self.sum_dim) \
                - 0.5 * torch.sum(torch.pow(eps, 2), dim=self.sum_dim)
        return z, log_p

    def log_prob(self, z, y=None):
        # Perpare parameter
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(self.log_scale_logs * self.logscale_factor)
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.loc.dtype,
                                       device=self.loc.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(y.size(0), self.shape[0],
                                               *((self.n_dim - 1) * [1]))
            log_scale = log_scale + (y @ self.log_scale_cc).view(y.size(0), self.shape[0],
                                                                 *((self.n_dim - 1) * [1]))
        if self.temperature is not None:
            log_scale = log_scale + np.log(self.temperature)
        # Get log prob
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - self.num_pix * torch.sum(log_scale, dim=self.sum_dim)\
                - 0.5 * torch.sum(torch.pow((z - loc) / torch.exp(log_scale), 2),
                                  dim=self.sum_dim)
        return log_p


class AffineGaussian(BaseDistribution):
    """
    Diagonal Gaussian an affine constant transformation applied to it,
    can be class conditional or not
    """
    def __init__(self, shape, affine_shape, num_classes=None):
        """
        Constructor
        :param shape: Shape of the variables
        :param affine_shape: Shape of the parameters in the affine transformation
        :param num_classes: Number of classes if the base is class conditional,
        None otherwise
        """
        super().__init__()
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.affine_shape = affine_shape
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        # Affine transformation
        if self.class_cond:
            self.transform = flows.CCAffineConst(self.affine_shape, self.num_classes)
        else:
            self.transform = flows.AffineConstFlow(self.affine_shape)
        # Temperature parameter for annealed sampling
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        dtype = self.transform.s.dtype
        device = self.transform.s.device
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=device)
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=dtype, device=device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        if self.temperature is not None:
            log_scale = np.log(self.temperature)
        else:
            log_scale = 0.
        # Sample
        eps = torch.randn((num_samples,) + self.shape, dtype=dtype, device=device)
        z = np.exp(log_scale) * eps
        # Get log prob
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - self.d * log_scale \
                - 0.5 * torch.sum(torch.pow(eps, 2), dim=self.sum_dim)
        # Apply transform
        if self.class_cond:
            z, log_det = self.transform(z, y)
        else:
            z, log_det = self.transform(z)
        log_p -= log_det
        return z, log_p

    def log_prob(self, z, y=None):
        # Perpare onehot encoding of class if needed
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes),
                                       dtype=self.transform.s.dtype,
                                       device=self.transform.s.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        if self.temperature is not None:
            log_scale = np.log(self.temperature)
        else:
            log_scale = 0.
        # Get log prob
        if self.class_cond:
            z, log_p = self.transform.inverse(z, y)
        else:
            z, log_p = self.transform.inverse(z)
        z = z / np.exp(log_scale)
        log_p = log_p - self.d * log_scale \
                - 0.5 * self.d * np.log(2 * np.pi) \
                - 0.5 * torch.sum(torch.pow(z, 2), dim=self.sum_dim)
        return log_p


class GaussianMixture(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """
    def __init__(self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True):
        """
        Constructor
        :param n_modes: Number of modes of the mixture model
        :param dim: Number of dimensions of each Gaussian
        :param loc: List of mean values
        :param scale: List of diagonals of the covariance matrices
        :param weights: List of mode probabilities
        :param trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim

        if loc is None:
            #a = np.linspace(-10, 10, self.n_modes)
            #loc = np.random.randn(self.n_modes, self.dim)#np.vstack([a]*4).T
            #loc = np.vstack([a] * self.dim).T
            loc = np.zeros((self.n_modes, self.dim))
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes) + 1.
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1. * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1. * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1. * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1. * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1. * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1. * weights)))

    def forward(self, num_samples=1):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Sample mode indices
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]

        # Get samples
        eps_ = torch.randn(num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device)
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = - 0.5 * self.dim * np.log(2 * np.pi) + torch.log(weights)\
                - 0.5 * torch.sum(torch.pow(eps, 2), 2) \
                - torch.sum(self.log_scale, 2)
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = - 0.5 * self.dim * np.log(2 * np.pi) + torch.log(weights) \
                - 0.5 * torch.sum(torch.pow(eps, 2), 2) \
                - torch.sum(self.log_scale, 2)
        log_p = torch.logsumexp(log_p, 1)

        return log_p


class GaussianPCA(BaseDistribution):
    """
    Gaussian distribution resulting from linearly mapping a normal distributed latent
    variable describing the "content of the target"
    """
    def __init__(self, dim, latent_dim=None, sigma=0.1):
        """
        Constructor
        :param dim: Number of dimensions of the flow variables
        :param latent_dim: Number of dimensions of the latent "content" variable;
                           if None it is set equal to dim
        :param sigma: Noise level
        """
        super().__init__()

        self.dim = dim
        if latent_dim is None:
            self.latent_dim = dim
        else:
            self.latent_dim = latent_dim

        self.loc = nn.Parameter(torch.zeros(1, dim))
        self.W = nn.Parameter(torch.randn(latent_dim, dim))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))

    def forward(self, num_samples=1):
        eps = torch.randn(num_samples, self.latent_dim, dtype=self.loc.dtype,
                          device=self.loc.device)
        z_ = torch.matmul(eps, self.W)
        z = z_ + self.loc

        Sig = torch.matmul(self.W.T, self.W) \
              + torch.exp(self.log_sigma * 2) \
              * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = self.dim / 2 * np.log(2 * np.pi) - 0.5 * torch.det(Sig) \
                - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)

        return z, log_p

    def log_prob(self, z):
        z_ = z - self.loc

        Sig = torch.matmul(self.W.T, self.W) \
              + torch.exp(self.log_sigma * 2) \
              * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = self.dim / 2 * np.log(2 * np.pi) - 0.5 * torch.det(Sig) \
                - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)

        return log_p
