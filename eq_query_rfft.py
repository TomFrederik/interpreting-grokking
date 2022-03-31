import os 

import einops
import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.fft import rfft, irfft, rfftfreq
from tqdm import tqdm

from datasets import get_dataset
from model import GrokkingTransformer
from utils import load_model
from scipy import signal

_NEXT_FAST_LEN = {}

def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Implementation from pyro

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1
        
def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension ``dim``.

    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    Implementation copied form `pyro <https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/stats.py>`_.

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of ``input``.
    """
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    # centering and padding x
    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(range(N, 0, -1), dtype=input.dtype, device=input.device)
    autocorr = autocorr / autocorr[..., :1]
    return autocorr.transpose(dim, -1)

def autocorr(x):
    outsize = x.shape[0]
    # circular padding
    x_pad = np.pad(x, (0, outsize), 'wrap')
    assert x_pad.shape[0] == 2 * outsize
    result = np.correlate(x, x_pad, mode='full')
    return result[outsize-1:2*outsize-1]


model_name = "Single Layer ReLU"
_, ckpt_dir = load_model(model_name)
paths = [ckpt_dir + f"/epoch={epoch}-step={epoch*10+9}.ckpt" for epoch in range(0,1071,5)]

os.makedirs(f'{model_name}/eq_query_rffts', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = torch.from_numpy(get_dataset('minus', 97, './data').data).to(device)[:,:-1]

for path in tqdm(paths[:-1]):
    epoch = int(path.split('/')[-1].split('-')[0].split('=')[-1])
    # if os.path.exists(f"{model_name}/eq_query_plots/epoch={epoch}.jpg"):
    #     continue
    model = GrokkingTransformer.load_from_checkpoint(path).to(device)
    model.eval()
    
    embedded_data = model.pos_encoding(model.embedding(dataset))
    qkv = model.transformer[0].self_attn.qkv_proj(embedded_data)
    qkv = einops.rearrange(qkv, 'batch seq_length (num_heads head_dim) -> batch num_heads seq_length head_dim', head_dim=3*model.transformer[0].self_attn.head_dim)
    q, k, v = qkv.chunk(3, dim=-1)
    equal_queries = q[:,:,-1]
    num_keys = k[:,:,[0,2]]
    
    
    dot_product = torch.einsum('bhn,bhtn->bht', equal_queries, num_keys)[[98*j for j in range(97)]].detach().cpu()
    dot_product = dot_product[...,0]
    # autocorr = autocorrelation(dot_product, dim=0)
    # print(autocorr)
    # print(autocorr.shape)
    # f, t, Pxx_den = signal.spectrogram(dot_product, axis=0, mode='magnitude')
    # print(f.shape)
    # print(Pxx_den.shape)
    dot_product -= dot_product.mean(dim=0, keepdim=True)
    dot_product /= dot_product.amax(dim=0, keepdim=True)
    
    autoc = []
    for h in range(4):
        autoc.append(autocorr(dot_product[:,h]))
    autoc = np.array(autoc).transpose(1,0)
    print(f"{autoc.shape = }")
    
    
    # periods = 1/rfftfreq(dot_product.shape[0])[1:]
    # periods = [p.item() for p in periods]
    # out = rfft(dot_product, dim=0, n=97, norm='ortho')
    # magnitudes = out.abs()[1:] # first element is "DC offset", i.e. 0 Hz
    # # print(magnitudes[:,1])
    # # raise ValueError
    fig, axes = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        # axes[i].bar(f, Pxx_den[:,i,0], label="First Num")
        # axes[i].bar(np.arange(48), magnitudes[:,i], label="First Num")
        axes[i].plot(np.arange(96), autoc[1:,i], label="First Num")
        # axes[i].set_xticks(np.arange(0,48,4), [f"{p:2.1f}" for i, p in enumerate(periods) if i%4==0])
        # axes[i].set_xticks(np.arange(0,49), (1/f).astype(np.int))
    axes[0].legend(bbox_to_anchor=(0.7, 1.05))
    plt.suptitle(f"Epoch {epoch}")
    plt.show()
    raise ValueError
    plt.savefig(f'{model_name}/eq_query_rffts/epoch={epoch}.jpg')
    plt.close()

images = []
fig = plt.figure()
for i in range(0,1071,5):
    image = mgimg.imread(f"{model_name}/eq_query_rffts/epoch={i}.jpg")
    images.append([plt.imshow(image)])
plt.axis('off')
my_anim = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=2000)
my_anim.save(f'{model_name}/eq_query_rfft_animation.gif', fps=10)