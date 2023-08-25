import torch
from torch import nn
from torch.distributions import Uniform, Normal

class DiscretisedBNF(nn.Module):

    def __init__(self, inDim, sigma1, K, hiddenDim):
        super().__init__()
        self.inDim = inDim
        self.sigma1 = torch.Tensor([sigma1]).cuda()
        self.K = K
        self.NN = nn.Sequential(
            nn.Linear(inDim + 1, hiddenDim),
            nn.LeakyReLU(),
            nn.Linear(hiddenDim, 2*inDim)
        )
        self.batched_discretised_cdfs = torch.vmap(self.discretised_cdfs, (1, 1, None))
        self.batched_discretised_cdf = torch.vmap(self.discretised_cdf, (None, None, 1))

    def discretised_cdf(self,mu, sigma, x):
        return 0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2)))))

    def discretised_cdfs(self,mu, sigma, x):
        return self.batched_discretised_cdf(mu,sigma,x)


    def discretised_output_distribution(self, mu, t, gamma, tmin=1e-10):

        D = mu.shape[1]

        mu_epsilon, ln_sigma_epsilon = torch.split(self.NN(torch.cat([mu, t],dim=1)), self.inDim, dim=1)

        mu_x = (mu/gamma) - torch.sqrt((1-gamma)/(gamma)) * (mu_epsilon)
        sigma_x = torch.sqrt((1-gamma)/(gamma))*torch.exp(ln_sigma_epsilon)
        if (t < tmin).any():
            for i, label in enumerate(t < tmin):
                if label == True:
                    mu_x[i,:] = torch.zeros_like(mu[0])
                    sigma_x[i,:] = torch.ones_like(mu[0])

        pO = []

        k_values = torch.arange(self.K, dtype=torch.float32, device='cuda').unsqueeze(0)  # Change 'cuda' to your desired device
        kl = (2 * (k_values - 1) / self.K) - 1
        kr = (2 * k_values / self.K) - 1
        kc = ((2 * k_values - 1) / self.K) - 1

        cdf_kl = self.batched_discretised_cdfs(mu_x, sigma_x, kl)
        cdf_kr = self.batched_discretised_cdfs(mu_x, sigma_x, kr)
        cdf_kl[:, (kl <= -1).squeeze(), :] = torch.Tensor([0]).cuda()
        cdf_kr[:,(kr<=-1).squeeze(),:] = torch.Tensor([0]).cuda()
        cdf_kl[:, (kl >= 1).squeeze(), :] = torch.Tensor([1]).cuda()
        cdf_kr[:,(kr >= 1).squeeze(),:] = torch.Tensor([1]).cuda()
        pO = kc * (cdf_kr - cdf_kl).transpose(1,2)
        pO = torch.sum(pO,2).T

        return pO

    def discretised_posterior(self, x,t):
        gamma = 1 - self.sigma1 ** (2 * t)
        mu = Normal(gamma * x, gamma * (1 - gamma)).sample()
        pO = self.discretised_output_distribution(mu, t, gamma)
        return pO

    def forward(self,x):
        t = Uniform(0, 1).sample((x.shape[0], 1)).cuda()
        pO = self.discretised_posterior(x,t)
        loss = -1 * torch.log(self.sigma1) * torch.mean((self.sigma1**(-2*t)) * torch.nn.functional.mse_loss(x,pO, reduce=False))
        return loss

    def iterative_sampling_process(self,inDim=100 , n=100):
        self.eval()
        mu = torch.zeros((1,inDim)).cuda()
        rho = torch.ones((1,inDim)).cuda()
        for i in range(1, n + 1):
            t = torch.ones((1,1)).cuda()*((i - 1) / n)
            k = self.discretised_output_distribution(mu,t,1-(self.sigma1**(2*t)))
            alpha = (self.sigma1 ** -2 * (i / n)) * (1 - self.sigma1 ** (2 / n))
            y = Normal(k, 1/alpha).sample()
            mu = ((rho*mu)+(alpha*y))/(rho+alpha)
            rho = rho + alpha
        k = self.discretised_output_distribution(mu, torch.ones((1,1)).cuda(), 1 - (self.sigma1 ** (2 * t)))
        return k
