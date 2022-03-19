import logging
import os

import einops
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.image as mgimg
import torch.nn.functional as F
import torch
from tqdm import tqdm
from unseal.hooks import HookedModel, Hook
from unseal.hooks.common_hooks import replace_activation

from datasets import ArithmeticData
from model import GrokkingTransformer
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    logging.warning('No GPU found! Using CPU')


ckpt, ckpt_dir = load_model()

model = GrokkingTransformer.load_from_checkpoint(ckpt).to(device)
model = HookedModel(model)
model.eval()

topk = 10
ranked_kn_nn = [323, 297, 243, 448,  98,  93,  20, 223, 325, 486,  92,  13, 238, 129,
         327, 315, 265, 139, 108, 275, 343, 286, 356, 125,  39, 228, 335, 506,
           1, 460, 501, 186,  46, 220, 303,  60,  94, 305, 368, 433, 268, 405,
          95, 252, 427, 337, 185, 455, 281, 133, 359, 113, 316, 170,  74, 363,
         319, 104, 163, 339, 158, 293, 498, 438,  17, 345, 211, 491, 244, 311,
         431, 466, 227,  28, 440, 130, 147, 155, 458, 230, 276, 206, 367, 100,
          29,   9,  53,  99, 148, 387, 342, 199, 190, 439,  19, 348, 247, 145,
         333, 382, 449, 152, 419,   3, 236, 209, 137, 120, 162, 294, 346, 364,
         283, 215,  58, 258, 473, 302, 413, 415,   4, 189, 395, 203, 234, 187,
         102, 321, 371, 277, 436, 200, 255, 270, 250, 474, 214, 242, 260,  26,
         480,  64, 174,  65, 231, 397, 131, 173, 272,  14, 365, 462, 224, 112,
         360, 204, 464,  71, 497, 245, 347,   8, 349, 235, 510, 172, 300, 430,
         453, 454, 399, 383, 273, 372, 178, 161, 404, 374,  52, 151, 212, 123,
         482, 160, 494,  10,  86,  36,  79, 285, 177,  66, 167, 304, 280,   7,
         126, 470,  32,  67, 322, 475, 216, 388, 210,  76, 385,  12, 156, 308,
         490, 142, 362,  85, 511, 307, 269,  33, 256, 232, 330, 468, 467, 410,
         264, 279, 378, 213, 407, 485, 508, 428,  38,  50, 271, 284, 396, 310,
          87, 239, 257, 150, 489, 106, 424, 391, 320, 251, 253, 298, 502, 262,
         128,  45, 400,  31, 481, 171, 340,  35, 103, 488, 144,  69, 140,  21,
         435, 119, 421,   0, 117, 504, 181, 452, 394, 499, 484, 493, 274, 477,
         450, 180, 384, 168,  48, 143, 105, 503, 463, 176, 121, 116, 263, 166,
         132, 182, 169, 443, 107, 487, 314, 134, 261, 361, 217, 196, 355,  43,
         403, 267,  72, 101, 492, 423,  62, 306,   6, 478, 183, 301,  81, 229,
         507,  91, 179, 445, 225, 437,  97,  40, 197, 141, 249,  30,  84, 309,
         381, 159, 338, 218, 146, 221, 110, 341,  78, 351,  90, 465,  96, 226,
         191, 122,  34,  37,  44, 429, 354, 165,  83, 471, 266, 282, 412, 201,
         240, 192, 376, 202, 278,  54, 233, 469, 326, 353, 426, 379,  42, 505,
         456, 329, 417, 344, 248,  80, 332, 207, 292,  16, 495, 136, 288, 194,
         317, 434, 380, 291, 115, 324,   5,  82,  56,  22, 289, 259, 184, 219,
         392, 461, 312, 398, 208,  18, 390,  15, 198, 135,  77, 393,  89, 318,
          23, 408, 472,  25,  41, 154, 287,  11, 509, 175, 357,  73,  47,  68,
         109, 352, 222, 409, 418, 114, 358, 254, 193, 401, 369, 118, 295,  51,
         237, 350, 414, 416, 451, 496, 375, 138, 377, 447, 459, 296, 441, 422,
          57, 111, 188,  61, 444, 457, 366, 425, 205, 406, 331, 483, 336, 446,
         370, 500, 479, 328, 195, 442, 411, 149, 334,  27, 153,  70, 476, 241,
         373,  63, 402,  88, 432,   2, 124, 420,  59,  75, 290, 127,  49, 157,
         389, 246, 299, 164,  24,  55, 386, 313] # for 14 - 0 = 0
# ranked_kn_nn = [389, 246, 299, 164,  24,  55, 386, 313] # for 14 - 0 = 0
# ranked_kn_nn.reverse()

os.makedirs('suppressing_kn_nn_plots', exist_ok=True)

data = torch.from_numpy(ArithmeticData(data_dir='data', func_name="minus", prime=97).data).long().to(device)
x = data[:,:-1]
y = data[:,-1]

for k in tqdm(range(topk)):
    coarse_kn_nn = ranked_kn_nn[:k+1]
    indices = "...,["+",".join(str(i) for i in coarse_kn_nn)+"]"
    print(indices)
    replacement_tensor = torch.zeros((len(coarse_kn_nn)), device=device)
    predictions = model(x, hooks=[Hook('transformer->0->linear_net->0', replace_activation(indices, replacement_tensor), 'suppress')])[:,-1].softmax(dim=-1)
    predictions = predictions[np.arange(len(predictions)), y]
    predictions = einops.rearrange(predictions, '(h w) -> h w', h=97)

    plt.figure()
    plt.imshow(predictions.detach().cpu().numpy(), cmap='viridis', origin='lower')
    plt.xticks(np.arange(0,97,8))
    plt.yticks(np.arange(0,97,8))
    plt.colorbar()
    plt.title(f"Ablating top {k+1}")
    plt.savefig(f"suppressing_kn_nn_plots/ablating_top_{k+1}.jpg")
    plt.close()

images = []
fig = plt.figure()
for k in range(topk):
    image = mgimg.imread(f"suppressing_kn_nn_plots/ablating_top_{k+1}.jpg")
    images.append([plt.imshow(image)])
plt.axis('off')
my_anim = animation.ArtistAnimation(fig, images, interval=200, repeat_delay=2000)
my_anim.save('suppressing_kn_nn.gif', fps=2)