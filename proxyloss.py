import torch.nn as nn
import torch
import math

class ProxyLoss(nn.Module):

    def __init__(self,
                 noise_variance: float = 0.0001):
        super().__init__()
        self.noise_variance = noise_variance

    def forward(self,prefix):
        noise = torch.randn(prefix.shape, device=prefix.device) * math.sqrt(self.noise_variance)
        v_proxy = prefix + noise
        proxy_loss_value = torch.norm(prefix - v_proxy, p=2, dim=1)
        proxy_loss_value = 0.5*proxy_loss_value.pow(2)
        return proxy_loss_value.mean()


