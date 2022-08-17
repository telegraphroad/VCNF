import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from scipy.stats import multivariate_t, lognorm, cauchy, invweibull, pareto, chi2
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


class Target(nn.Module):
    """
    Sample target distributions to test models
    """
    def __init__(self, prop_scale=torch.tensor(20.),
                 prop_shift=torch.tensor(-10.)):
        """
        Constructor
        :param prop_scale: Scale for the uniform proposal
        :param prop_shift: Shift for the uniform proposal
        """
        super().__init__()
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        raise NotImplementedError('The log probability is not implemented yet.')

    def rejection_sampling(self, num_steps=1):
        """
        Perform rejection sampling on image distribution
        :param num_steps: Number of rejection sampling steps to perform
        :return: Accepted samples
        """
        eps = torch.rand((num_steps, self.n_dims), dtype=self.prop_scale.dtype,
                         device=self.prop_scale.device)
        z_ = self.prop_scale * eps + self.prop_shift
        prob = torch.rand(num_steps, dtype=self.prop_scale.dtype,
                          device=self.prop_scale.device)
        prob_ = torch.exp(self.log_prob(z_) - self.max_log_prob)
        #print('~~~',eps.device,z_.device,prob.device,prob_.device)
        with torch.no_grad():
            accept = prob_.cpu() > prob.cpu()
        z = z_[accept, :]
        return z

    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        z = torch.zeros((0, self.n_dims), dtype=self.prop_scale.dtype,
                        device=self.prop_scale.device)
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z

    
class NealsFunnel(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,prop_scale=torch.tensor(20.),
                 prop_shift=torch.tensor(-10.), v1shift = 0., v2shift = 0.):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.
        self.v1shift = v1shift
        self.v2shift = v2shift
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)


    # def sample(self, num_samples=1):
    #     """
    #     :param num_samples: Number of samples to draw
    #     :return: Samples
    #     """
    #     data = []
    #     n_dims = 1
    #     for i in range(nsamples):
    #         v = norm(0, 1).rvs(1)
    #         x = norm(0, np.exp(0.5*v)).rvs(n_dims)
    #         data.append(np.hstack([v, x]))
    #     data = pd.DataFrame(data)
    #     return torch.tensor(data.values)

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        #print('++++++++++',z)
        v = z[:,0].cpu()
        x = z[:,1].cpu()
        v_like = Normal(torch.tensor([0.0]).cpu(), torch.tensor([1.0]).cpu() + self.v1shift).log_prob(v).cpu()
        x_like = Normal(torch.tensor([0.0]).cpu(), torch.exp(0.5*v).cpu() + self.v2shift).log_prob(x).cpu()
        return v_like + x_like


class StudentTDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,df=2.,dim=2):
        super().__init__()
        self.df = df
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(multivariate_t(loc=self.loc,df=self.df).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(multivariate_t(loc=self.loc,df=self.df).logpdf(z.cpu().detach().numpy()),device='cuda')

class chi2Dist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,df=2.,dim=2):
        super().__init__()
        self.df = df
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(chi2(loc=self.loc,df=self.df).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(chi2(loc=self.loc,df=self.df).logpdf(z.cpu().detach().numpy()),device='cuda')


class FrechetDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,c=2.,dim=2):
        super().__init__()
        self.c = c
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(invweibull(loc=self.loc,c=self.c).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(invweibull(loc=self.loc,c=self.c).logpdf(z.cpu().detach().numpy()),device='cuda')


class ParetoDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,b=2.,dim=2):
        super().__init__()
        self.b = b
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(pareto(loc=self.loc,b=self.b).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(pareto(loc=self.loc,b=self.b).logpdf(z.cpu().detach().numpy()),device='cuda')


class CauchyDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,dim=2):
        super().__init__()
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(cauchy(loc=self.loc).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(cauchy(loc=self.loc).logpdf(z.cpu().detach().numpy()),device='cuda')


class LogNormDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,s=2.,dim=2):
        super().__init__()
        self.s = s
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(lognorm(loc=self.loc,s=self.s).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(lognorm(loc=self.loc,s=self.s).logpdf(z.cpu().detach().numpy()),device='cuda')

  
class TwoMoons(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.

    def log_prob(self, z):
        """
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        log_prob = - 0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2 \
                   - 0.5 * ((a - 2) / 0.3) ** 2 \
                   + torch.log(1 + torch.exp(-4 * a / 0.09))
        return log_prob


class CircularGaussianMixture(nn.Module):
    """
    Two-dimensional Gaussian mixture arranged in a circle
    """
    def __init__(self, n_modes=8):
        """
        Constructor
        :param n_modes: Number of modes
        """
        super(CircularGaussianMixture, self).__init__()
        self.n_modes = n_modes
        self.register_buffer("scale", torch.tensor(2 / 3 * np.sin(np.pi / self.n_modes)).float())

    def log_prob(self, z):
        d = torch.zeros((len(z), 0), dtype=z.dtype, device=z.device)
        for i in range(self.n_modes):
            d_ = ((z[:, 0] - 2 * np.sin(2 * np.pi / self.n_modes * i)) ** 2
                  + (z[:, 1] - 2 * np.cos(2 * np.pi / self.n_modes * i)) ** 2)\
                 / (2 * self.scale ** 2)
            d = torch.cat((d, d_[:, None]), 1)
        log_p = - torch.log(2 * np.pi * self.scale ** 2 * self.n_modes) \
                + torch.logsumexp(-d, 1)
        return log_p

    def sample(self, num_samples=1):
        eps = torch.randn((num_samples, 2), dtype=self.scale.dtype, device=self.scale.device)
        phi = 2 * np.pi / self.n_modes * torch.randint(0, self.n_modes, (num_samples,),
                                                       device=self.scale.device)
        loc = torch.stack((2 * torch.sin(phi), 2 * torch.cos(phi)), 1).type(eps.dtype)
        return eps * self.scale + loc


class RingMixture(Target):
    """
    Mixture of ring distributions in two dimensions
    """
    def __init__(self, n_rings=2):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.
        self.n_rings = n_rings
        self.scale = 1 / 4 / self.n_rings

    def log_prob(self, z):
        d = torch.zeros((len(z), 0), dtype=z.dtype, device=z.device)
        for i in range(self.n_rings):
            d_ = ((torch.norm(z, dim=1) - 2 / self.n_rings * (i + 1)) ** 2) \
                 / (2 * self.scale ** 2)
            d = torch.cat((d, d_[:, None]), 1)
        return torch.logsumexp(-d, 1)






# Modified from https://github.com/jhjacobsen/invertible-resnet/blob/278faffe7bf25cd7488f8cd49bf5c90a1a82fc0c/models/toy_data.py#L8 
def get_2d_data(data, size):
    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=size, noise=1.0)[0]
        data = data[:, [0, 2]]
        data /= 5

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=size, factor=.5, noise=0.08)[0]
        data *= 3

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = size // 4
        n_samples1 = size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        # Add noise
        data = X + np.random.normal(scale=0.08, size=X.shape)

    elif data == "8gaussians":
        dim = 2
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        for i in range(len(centers)):
          for k in range(dim-2):
            centers[i] = centers[i]+(0,)

        data = []
        for i in range(size):
            point = np.random.randn(dim) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            data.append(point)
        data = np.array(data)
        data /= 1.414

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        data = x

    elif data == "checkerboard":
        x1 = np.random.rand(size) * 4 - 2
        x2_ = np.random.rand(size) - np.random.randint(0, 2, size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1)
        data *= 2

    elif data == "line":
        x = np.random.rand(size) * 5 - 2.5
        y = x
        data = np.stack((x, y), 1)
    elif data == "cos":
        x = np.random.rand(size) * 5 - 2.5
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)

    elif data == "2uniforms":
        mixture_component = (np.random.rand(size) > 0.5).astype(int)
        x1 = np.random.rand(size) + mixture_component - 2*(1 - mixture_component)
        x2 = 2 * (np.random.rand(size) - 0.5)
        data = np.stack((x1, x2), 1)

    elif data == "2lines":
        x1 = np.empty(size)
        x1[:size//2] = -1.
        x1[size//2:] = 1.
        x1 += 0.01 * (np.random.rand(size) - .5)
        x2 = 2 * (np.random.rand(size) - 0.5)
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "2marginals":
        x1 = np.empty(size)
        x1[:size//2] = -1.
        x1[size//2:] = 1.
        x1 += .5 * (np.random.rand(size) - .5)
        x2 = np.random.normal(size=size)
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "1uniform":
        x1 = np.random.rand(size) - .5
        x2 = np.random.rand(size) - .5
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "annulus":
        rad1 = 2
        rad2 = 1
        theta = 2 * np.pi * np.random.random(size)
        r = np.sqrt(np.random.random(size) * (rad1**2 - rad2**2) + rad2**2)
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        data = np.stack((x1, x2), 1)

    elif data == "sawtooth":
        u = np.random.rand(size)
        branch = u < .5
        x1 = np.zeros(size)
        x1[branch] = -1 - np.sqrt(1 - 2*u[branch])
        x1[~branch] = 1 + np.sqrt(2*u[~branch] - 1)
        x2 = np.random.rand(size)
        data = np.stack((x1, x2), 1)

    elif data == "quadspline":
        u = np.random.rand(size)
        branch = u < .5
        x1 = np.zeros(size)
        x1[branch] = -1 + np.cbrt(2*u[branch] - 1)
        x1[~branch] = 1 + np.cbrt(2*u[~branch] - 1)
        x2 = np.random.rand(size)
        data = np.stack((x1, x2), 1)

    elif data == "split-gaussian":
        x1 = np.random.normal(size=size)
        x2 = np.random.normal(size=size)
        x2[x1 >= 0] += 2
        x2[x1 < 0] -= 2
        data = np.stack((x1, x2), 1)

    else:
        assert False, f"Unknown dataset `{data}''"

    return torch.tensor(data, dtype=torch.get_default_dtype())
