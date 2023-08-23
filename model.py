import torch
from torch import nn
from torch.distributions import Uniform, Normal

class DiscretisedBNF(nn.Module):

    def __init__(self, inDim, sigma1, K, hiddenDim):
        super().__init__()
        self.inDim = inDim
        self.sigma1 = sigma1
        self.K = K
        self.NN = nn.Sequential(
            nn.Linear(inDim + 1, hiddenDim),
            nn.LeakyReLU(),
            nn.Linear(hiddenDim, 2*inDim)
        )

    def discretised_cdf(self,mu, sigma, x):
        F = 0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2)))))
        if x<=-1:
            G = torch.Tensor([0])
        elif x>=1:
            G = torch.Tensor([1])
        else:
            G = F
        return G

    import torch
    from torch import nn

    def discretised_output_distribution(self, mu, t, gamma, tmin=1e-10):

        D = mu.shape[1]

        if t < tmin:
            mu_x = torch.zeros_like(mu)
            sigma_x = torch.ones_like(mu)
        else:
            # Run network to get mu_epsilon and ln(sigma_epsilon)
            mu_epsilon, ln_sigma_epsilon = torch.split(self.NN(torch.cat([mu, t],dim=1)), self.inDim, dim=1)

            mu_x = (mu/gamma) - torch.sqrt((1-gamma)/(gamma)) * (mu_epsilon)
            sigma_x = torch.sqrt((1-gamma)/(gamma))*torch.exp(ln_sigma_epsilon)

        pO = []

        for d in range(D):
            p_temp = []
            for k in range(self.K):
                kl = (2 * (k - 1) / self.K) - 1
                kr = (2 * k / self.K) - 1
                cdf_kl = self.discretised_cdf(mu_x[:,d], sigma_x[:,d], kl)
                cdf_kr = self.discretised_cdf(mu_x[:,d], sigma_x[:,d], kr)
                p_temp.append(cdf_kr - cdf_kl)
            pO.append(torch.stack(p_temp))

        return torch.sum(torch.stack(pO),1).T

    def discretised_posterior(self, x):
        t = Uniform(0, 1).sample((x.shape[0],1))
        gamma = 1 - self.sigma1 ** 2 * t
        mu = Normal(gamma * x, gamma * (1 - gamma)).sample()
        pO = self.discretised_output_distribution(mu, t, gamma)
        return pO

    def iterative_sampling_process(self,inDim=100 , n=100):
        mu = torch.zeros((1,inDim))
        rho = torch.ones((1,inDim))
        for i in range(1, n + 1):
            t = torch.ones((1,1))*((i - 1) / n)
            k = self.discretised_output_distribution(mu,t,1-(self.sigma1**(2*t)))
            alpha = self.sigma1 ** -2 * (i / n) * (1 - self.sigma1 ** (2 / n))
            y = Normal(k, 1/alpha).sample()
            mu = ((rho*mu)+(alpha*y))/(rho+alpha)
            rho = rho + alpha
        k = self.discretised_output_distribution(mu, torch.ones((1,1)), 1 - (self.sigma1 ** (2 * t)))
        return k
model = DiscretisedBNF(inDim=100 ,sigma1=0.5, K=10, hiddenDim= 16)

x = torch.rand((1,100))
posterior = model.discretised_posterior(x)

k = model.iterative_sampling_process(inDim=100 ,n=100)
print(k)