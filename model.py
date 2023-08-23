import torch
from torch import nn
from torch.distributions import Uniform, Normal

class DiscretisedDiffusion(nn.Module):

    def __init__(self, sigma1, K, hiddenDim):
        super().__init__()
        self.sigma1 = sigma1
        self.K = K
        self.NN = nn.Sequential(
            nn.Linear(2*K + 1, hiddenDim),
            nn.LeakyReLU(),
            nn.Linear(hiddenDim, 2*K)
        )

    def discretised_cdf(self,mu, sigma, x):
        F = 0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2)))))
        G = torch.where(x <= -1, torch.tensor(0),
                        torch.where(x >= 1, torch.tensor(1), F))
        return G

    import torch
    from torch import nn

    def discretised_output_distribution(self, mu, t, K, gamma, tmin=1e-10):

        D = mu.shape[0]

        if t < tmin:
            mu_x = torch.zeros_like(mu)
            sigma_x = torch.ones_like(mu)
        else:
            # Run network to get mu_epsilon and ln(sigma_epsilon)
            mu_epsilon, ln_sigma_epsilon = self.NN(torch.cat([mu, t]))

            mu_x = mu ** (gamma - 1 / (1 - gamma)) * (mu_epsilon ** (1 / (1 - gamma)))
            sigma_x = (1 - gamma) ** (-1 / 2) * torch.exp(ln_sigma_epsilon / 2)

        pO = []
        for d in range(D):
            pO.append([])
            for k in range(K):
                kl = 2 * (k - 1) / (K - 1)
                kr = 2 * k / (K - 1)
                cdf_kl = self.discretised_cdf(mu_x[d], sigma_x[d], kl)
                cdf_kr = self.discretised_cdf(mu_x[d], sigma_x[d], kr)
                pO[d].append(cdf_kr - cdf_kl)

        return torch.stack(pO)

    def discretised_posterior(self, x):
        t = Uniform(0, 1).sample()
        gamma = 1 - self.sigma1 ** 2 * t
        mu = Normal(gamma * x, gamma * (1 - gamma)).sample()
        pO = self.discretised_output_distribution(mu, t, gamma)
        return pO

    def iterative_process(self, n):
        mu = torch.zeros(1)
        rho = torch.ones(1)
        for i in range(1, n + 1):
            t = (i - 1) / n
            gamma = 1 - self.sigma1 ** 2 * t
            k = self.discretised_posterior(mu)
            alpha = self.sigma1 ** -2 * (i / n) * (1 - self.sigma1 ** 2 / n)
            y = Normal(k, alpha).sample()
            mu = (i - 1) / i * mu + y / i
            rho = self.sigma1 ** 2 * (1 - 1 / i ** 2)
        return mu, rho

model = DiscretisedDiffusion(sigma1=0.5, K=10)

x = torch.rand(100)
posterior = model.discretised_posterior(x)

mu, rho = model.iterative_process(n=100)