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

    def discretised_cdf(self,mu, sigma, x):
        F = 0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2)))))
        if x<=-1:
            G = torch.zeros_like(F,requires_grad=False).cuda()
        elif x>=1:
            G = torch.ones_like(F,requires_grad=False).cuda()
        else:
            G = F
        return G

    import torch
    from torch import nn

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

        for d in range(D):
            p_temp = []
            for k in range(self.K):
                kl = (2 * (k - 1) / self.K) - 1
                kr = (2 * k / self.K) - 1
                kc = ((2*k-1)/self.K)-1
                cdf_kl = self.discretised_cdf(mu_x[:,d], sigma_x[:,d], kl)
                cdf_kr = self.discretised_cdf(mu_x[:,d], sigma_x[:,d], kr)
                p_temp.append(kc*(cdf_kr - cdf_kl))
            pO.append(torch.stack(p_temp))

        return torch.sum(torch.stack(pO),1).T

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
        for i in range(2, n + 1):
            t = torch.ones((1,1)).cuda()*((i - 1) / n)
            k = self.discretised_output_distribution(mu,t,1-(self.sigma1**(2*t)))
            alpha = self.sigma1 ** -2 * (i / n) * (1 - self.sigma1 ** (2 / n))
            y = Normal(k, 1/alpha).sample()
            mu = ((rho*mu)+(alpha*y))/(rho+alpha)
            rho = rho + alpha
        k = self.discretised_output_distribution(mu, torch.ones((1,1)).cuda(), 1 - (self.sigma1 ** (2 * t)))
        return k

# model = DiscretisedBNF(inDim=100 ,sigma1=0.5, K=10, hiddenDim= 16)
#
# num_signals = 1000
# signal_length = 100
#
# # Generate random frequencies and amplitudes for the signals
# frequencies = torch.rand(num_signals) * 10  # Random frequencies between 0 and 10
# amplitudes = torch.rand(num_signals) * 2    # Random amplitudes between 0 and 2
#
# # Create a time vector
# t = torch.linspace(0, 2 * np.pi, signal_length)
#
# # Initialize an empty tensor to store the signals
# x = torch.zeros(num_signals, signal_length)
# for i in range(num_signals):
#     x[i, :] = amplitudes[i] * torch.sin(frequencies[i] * t)
#
# posterior = model.discretised_posterior(x)
#
# k = model.iterative_sampling_process(inDim=100 ,n=100)
# print(k)