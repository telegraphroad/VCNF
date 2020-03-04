import torch
import torch.nn as nn
import numpy as np

class ParametrizedConditionalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        """
        :param x: Variable to condition on, first dimension is batch size
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        raise NotImplementedError

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch size
        :param x: Variable to condition on, first dimension is batch size
        :return: log probability of z given x
        """
        raise NotImplementedError


class Dirac(ParametrizedConditionalDistribution):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        z = x.unsqueeze(1).repeat(1, num_samples, 1)
        log_p = torch.zeros(z.size()[0:2])
        return z, log_p

    def log_prob(self, z, x):
        log_p = torch.zeros(z.size()[0:2])
        return log_p


class ConstDiagGaussian(ParametrizedConditionalDistribution):
    def __init__(self, loc, scale):
        """
        Multivariate Gaussian distribution with diagonal covariance and parameters being constant wrt x
        :param loc: mean vector of the distribution
        :param scale: vector of the standard deviations on the diagonal of the covariance matrix
        """
        super().__init__()
        self.n = len(loc)
        if not torch.is_tensor(loc):
            loc = torch.tensor(loc)
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale)
        self.loc = nn.Parameter(loc.reshape((1, 1, self.n)))
        self.scale = nn.Parameter(scale)

    def forward(self, x=None, num_samples=1):
        """
        :param x: Variable to condition on, will only be used to determine the batch size
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        if x is not None:
            batch_size = len(x)
        else:
            batch_size = 1
        eps = torch.randn((batch_size, num_samples, self.n), device=x.device)
        z = self.loc + self.scale * eps
        log_p = - 0.5 * self.n * np.log(2 * np.pi) - torch.sum(torch.log(self.scale) + 0.5 * torch.pow(eps, 2), 2)
        return z, log_p

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch dimension
        :param x: Variable to condition on, first dimension is batch dimension
        :return: log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        log_p = - 0.5 * self.n * np.log(2 * np.pi) - torch.sum(torch.log(self.scale) + 0.5 * ((z - self.loc) / self.scale) ** 2, 2)
        return log_p


class NNDiagGaussian(ParametrizedConditionalDistribution):
    """
    Diagonal Gaussian distribution with mean and variance determined by a neural network
    """
    def __init__(self, net):
        """
        Constructor
        :param net: net computing mean (first n / 2 outputs), standard deviation (second n / 2 outputs)
        """
        self.net = net

    def forward(self, x, num_samples=1):
        """
        :param x: Variable to condition on
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        batch_size = len(x)
        eps = torch.randn((batch_size, num_samples) + (x.dim() - 1) * (1,), device=x.device)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[2] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        std = mean_std[:, n_hidden:(2 * n_hidden), ...].unsqueeze(1)
        z = mean + std * eps
        log_p = - 0.5 * torch.prod(z.size()[2:]) * np.log(2 * np.pi)\
                - torch.sum(torch.log(std) + 0.5 * torch.pow(eps, 2), list(range(2, self.z.dim())))
        return z, log_p

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch dimension
        :param x: Variable to condition on, first dimension is batch dimension
        :return: log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[2] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        std = mean_std[:, n_hidden:(2 * n_hidden), ...].unsqueeze(1)
        log_p = - 0.5 * torch.prod(z.size()[2:]) * np.log(2 * np.pi)\
                - torch.sum(torch.log(std) + 0.5 * ((z - mean) / std) ** 2, 2)
        return log_p


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        Decodes z to x
        :param z: latent variable
        :return: x, std of x
        """
        raise NotImplementedError

    def log_p(self, x, z):
        """
        :param x: observable
        :param z: latent variable
        :return: log(p) of x given z
        """
        raise NotImplementedError


class NNDiagGaussianDecoder(Decoder):
    """
    Decoder representing a diagonal Gaussian distribution with mean and std parametrized by a NN
    """
    def __init__(self, net):
        """
        Constructor
        :param net: neural network parametrizing mean and standard deviation of diagonal Gaussian
        """
        self.net = net

    def forward(self, z):
        mean_std = self.net(z)
        n_hidden = mean_std.size()[2] // 2
        mean = mean_std[:, :n_hidden, ...]
        std = mean_std[:, n_hidden:(2 * n_hidden), ...]
        return mean, std

    def log_p(self, x, z):
        mean_std = self.net(z)
        n_hidden = mean_std.size()[2] // 2
        mean = mean_std[:, :n_hidden, ...]
        std = mean_std[:, n_hidden:(2 * n_hidden), ...]
        log_p = - 0.5 * torch.prod(z.size()[2:]) * np.log(2 * np.pi) \
                - torch.sum(torch.log(std) + 0.5 * ((z - mean) / std) ** 2, list(range(1, self.z.dim())))
        return log_p


class PriorDistribution:
    def __init__(self):
        raise NotImplementedError

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        raise NotImplementedError


class TwoModes(PriorDistribution):
    def __init__(self, loc, scale):
        """
        Distribution 2d with two modes at z[0] = -loc and z[0] = loc
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.loc = loc
        self.scale = scale

    def log_prob(self, z):
        """
        log(p) = 1/2 * ((norm(z) - loc) / (2 * scale)) ** 2
                - log(exp(-1/2 * ((z[0] - loc) / (3 * scale)) ** 2) + exp(-1/2 * ((z[0] + loc) / (3 * scale)) ** 2))
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
        log_prob = - 0.5 * ((torch.norm(z_, dim=0) - self.loc) / (2 * self.scale)) ** 2\
                   + torch.log(torch.exp(-0.5 * ((z_[0] - self.loc) / (3 * self.scale)) ** 2)
                               + torch.exp(-0.5 * ((z_[0] + self.loc) / (3 * self.scale)) ** 2))
        return log_prob
    
    
class Sinusoidal(PriorDistribution):
    def __init__(self, scale, period):
        """
        Distribution 2d with sinusoidal density
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.scale = scale
        self.period = period

    def log_prob(self, z):
        """
        log(p) = 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        w_1(z) = sin(2*pi / period * z[0])
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
            
        w_1 = lambda x: torch.sin(2*np.pi / self.period * z_[0])
        log_prob = - 0.5 * ((z_[1] - w_1(z_)) / (self.scale)) ** 2
        
        return log_prob
    
    
class Sinusoidal_gap(PriorDistribution):
    def __init__(self, scale, period):
        """
        Distribution 2d with sinusoidal density
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.scale = scale
        self.period = period
        self.w2_scale = 0.6
        self.w2_amp = 3.0
        self.w2_mu = 1.0

    def log_prob(self, z):
        """
        log(p) = 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        w_1(z) = sin(2*pi / period * z[0])
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
            
        w_1 = lambda x: torch.sin(2*np.pi / self.period * z_[0])
        w_2 = lambda x: self.w2_amp * torch.exp(-0.5*((z_[0] - self.w2_mu) / self.w2_scale)**2)
        log_prob = torch.log( torch.exp(-0.5 * ((z_[1] - w_1(z_)) / (self.scale)) ** 2 ) + \
                             torch.exp(-0.5 * ((z_[1] - w_1(z_) + w_2(z_)) / (self.scale)) ** 2 ) )
        
        return log_prob
    
    
class Sinusoidal_split(PriorDistribution):
    def __init__(self, scale, period):
        """
        Distribution 2d with sinusoidal density
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.scale = scale
        self.period = period
        self.w3_scale = 0.3
        self.w3_amp = 3.0
        self.w3_mu = 1.0

    def log_prob(self, z):
        """
        log(p) = 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        w_1(z) = sin(2*pi / period * z[0])
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
            
        w_1 = lambda x: torch.sin(2*np.pi / self.period * z_[0])
        w_3 = lambda x: self.w3_amp * torch.sigmoid((z_[0] - self.w3_mu) / self.w3_scale)
        log_prob = torch.log( torch.exp(-0.5 * ((z_[1] - w_1(z_)) / (self.scale)) ** 2 ) + \
                             torch.exp(-0.5 * ((z_[1] - w_1(z_) + w_3(z_)) / (self.scale)) ** 2 ) )
        
        return log_prob
    
    
class Smiley(PriorDistribution):
    def __init__(self, scale):
        """
        Distribution 2d with sinusoidal density
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.scale = scale
        self.loc = 2.0

    def log_prob(self, z):
        """
        log(p) = 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        w_1(z) = sin(2*pi / period * z[0])
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
            
        log_prob = - 0.5 * ((torch.norm(z_, dim=0) - self.loc) / (2 * self.scale)) ** 2\
                   - 0.5 * ((torch.abs(z_[1] + 0.8) - 1.2) / (2 * self.scale)) ** 2
        
        return log_prob