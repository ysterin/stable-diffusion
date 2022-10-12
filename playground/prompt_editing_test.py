import torch
import numpy as np
import k_diffusion as K
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from scripts.img2img import load_img
from scripts.txt2img import load_model_from_config, chunk
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from matplotlib import pyplot as plt
from img2noise import find_noise_for_image, find_noise_for_latent, pil_img_to_latent, pil_img_to_torch

from PIL import Image
from torch import autocast
from einops import rearrange, repeat
from tqdm import trange
import os

from utils import CFGCompVisDenoiser, get_progress_bar

seed = 407
n_iter = 1
batch_size = 1
seed_everything(seed)
H, W, C, f = 512, 512, 4, 8

# prompt = "Photo of a fashion model with long hair, studio lightning"
# reversion_prompt = "Photo of a fashion model with short hair, studio lightning"


# reversion_prompt = prompt
save_dir = "../outputs/edit_test/7"
scale = 2
inversion_scale = 2
reversion_scale = 2
plms = False
ddim_steps = 50
inversion_steps = 50
reversion_steps = 50
ddim_eta = 0.0

config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "../models/ldm/stable-diffusion-v1/model.ckpt")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

prompt = "Photo of a smiling woman with lob hairstyle hair"
reversion_prompt = "Photo of a smiling woman with crew cut hairstyle hair"

uc = model.get_learned_conditioning(batch_size * [""])
cond = model.get_learned_conditioning(batch_size * [prompt])
reversion_cond = model.get_learned_conditioning(batch_size * [reversion_prompt])
reversion_prompt1 = model.get_learned_conditioning(batch_size * ["crew cut hairstyle"])
lmda = 0.001
# reversion_cond = lmda * reversion_cond + (1 - lmda) * reversion_prompt1
reversion_cond = 1 * reversion_cond + 1 * reversion_prompt1

# reversion_cond[:, 7:9] *= 1.5

image_path = "../assets/sample_images/faces/girl_head.png"

image = Image.open(image_path)

image = image.resize(size=(H, W))

init_image = load_img(image_path, size=(H, W)).to(device)
init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
# init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
# decoded_latent_image = model.decode_first_stage(init_latent)
# decoded_latent_image = rearrange(decoded_latent_image, 'b c h w -> b h w c')
# decoded_latent_image = decoded_latent_image.cpu().numpy()
# decoded_latent_image = np.clip((decoded_latent_image + 1) / 2, 0, 1)
# decoded_latent_image = (decoded_latent_image * 255).astype(np.uint8)[0]

# dnw = K.external.CompVisDenoiser(model)
dnw = CFGCompVisDenoiser(model)
sigmas = dnw.get_sigmas(ddim_steps)

sample_fn = lambda *args, **kwargs: K.sampling.sample_euler(*args, **kwargs, s_noise=ddim_eta)
sample_fn = K.sampling.sample_lms

clip_tokenizer = model.cond_stage_model.tokenizer
def prompt_token(prompt, index):
    tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
    return clip_tokenizer.decode(tokens[index:index+1])

reverse_sigmas = dnw.get_sigmas(inversion_steps).flip(0)
reverse_sigmas[0] = 1e-5
scales = [0.5, 1.0, 1.5, 2.0, 2.5, 4.0, 5.0, 7.5, 10, 12.5, 15]
n_steps = 45
with torch.no_grad():
    with autocast("cuda", dtype=torch.float16):
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        for scale in scales[1:8]:
            inversion_scale = 0.1
            recon_noise = sample_fn(dnw, init_latent, reverse_sigmas[:n_steps],
                                    extra_args={'cond': cond, 'uncond': uc, 'cfg_scale': inversion_scale},
                                    callback=get_progress_bar(50, "invert sampling"))

            recon_sample = sample_fn(dnw, recon_noise, sigmas[-n_steps:],
                                     extra_args={'cond': reversion_cond, 'uncond': uc, 'cfg_scale': scale},
                                     callback=get_progress_bar(50, "sampling"))

            recon_image = model.decode_first_stage(recon_sample)
            recon_image = rearrange(recon_image, 'b c h w -> b h w c')

            recon_image = ((recon_image + 1) / 2).clamp(0, 1)
            recon_image = recon_image.cpu().numpy()
            recon_image = np.clip(recon_image * 255, 0, 255).astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image)
            # axes[1].imshow(decoded_latent_image)
            axes[2].imshow(recon_image[0])

            axes[0].set_title("Original")
            # axes[1].set_title("Decoded Latent")
            axes[2].set_title("Reconstructed - scale: {}".format(scale))

            plt.show()




