import torch



class Logit():
    """
    Transform for dataloader
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    """
    def __init__(self, alpha=0):
        """
        Constructor
        :param alpha: see above
        """
        self.alpha = alpha

    def __call__(self, x):
        x_ = self.alpha + (1 - self.alpha) * x
        return torch.log(x_ / (1 - x_))

    def inverse(self, x):
        return (torch.sigmoid(x) - self.alpha) / (1 - self.alpha)


class Jitter():
    """
    Transform for dataloader
    Adds uniform jitter noise to data
    """
    def __init__(self, scale=1./256):
        """
        Constructor
        :param scale: Scaling factor for noise
        """
        self.scale = scale

    def __call__(self, x):
        eps = torch.rand_like(x) * self.scale
        x_ = x + eps
        return x_


class Scale():
    """
    Transform for dataloader
    Adds uniform jitter noise to data
    """
    def __init__(self, scale=255./256.):
        """
        Constructor
        :param scale: Scaling factor for noise
        """
        self.scale = scale

    def __call__(self, x):
        return x * self.scale