import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm.auto import tqdm

from model import DiscretisedBNF
torch.cuda.set_device(1)
def normalize_signal(signal, K):
    # Calculate the minimum and maximum values in the original signal
    # Normalize the signal to the desired range
    normalized_signal = ((2*signal-1)/K)-1
    return normalized_signal

def get_datapoint(batch=128, device='cpu'):
    num_signals = batch
    signal_length = 32

    # Generate random frequencies and amplitudes for the signals
    frequencies = 2 + torch.rand(num_signals) * 0.1  # Random frequencies between 0 and 10
    amplitudes = torch.rand(num_signals)  # Random amplitudes between 0 and 2

    # Create a time vector
    t = torch.linspace(0, 2 * np.pi, signal_length)

    # Initialize an empty tensor to store the signals
    X = torch.zeros(num_signals, signal_length)
    for i in range(num_signals):
        X[i, :] = amplitudes[i] * torch.sin(frequencies[i] * t)

    # X = torch.round(torch.nn.functional.relu(X))
    # X = normalize_signal(X, 16)
    # X = torch.round((X/X.max())*255)
    return X #torch.nn.functional.relu(X)

X = get_datapoint()  # (B, D=2) with K=2 classes

plt.title("Dataset")
plt.scatter(torch.linspace(0, 2 * np.pi, X.shape[1]), X[0, :]);
plt.grid()
plt.show()




bfn = DiscretisedBNF(inDim=X.shape[1] ,sigma1=0.001**0.5, K=16, hiddenDim= X.shape[1])
bfn.cuda()

optim = AdamW(bfn.parameters(), lr=1e-4)


n = 10000
losses = []
for i in tqdm(range(n)):
    optim.zero_grad()
    with torch.no_grad():
        X = get_datapoint().cuda()

    loss = bfn.forward(X)

    loss.backward()

    optim.step()

    losses.append(loss.item())

plt.plot(losses)
plt.show()


x_hat = bfn.iterative_sampling_process(inDim=X.shape[1] ,n=10).detach().cpu().numpy()

plt.title("Dataset")
plt.scatter(torch.linspace(0, 2 * np.pi, X.shape[1]), X[0, :].cpu().detach());
plt.scatter(torch.linspace(0, 2 * np.pi, X.shape[1]), x_hat[0, :]);
plt.grid()
plt.show()