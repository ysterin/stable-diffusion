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


class CFGCompVisDenoiser(K.external.CompVisDenoiser):
    def get_eps(self, x, t, cond, uncond=None, cfg_scale=1.0, **kwargs):
        if cfg_scale == 1.0:
            return self.inner_model.apply_model(x, t, cond=cond)
        if uncond is None:
            uncond = self.inner_model.get_learned_conditioning([""] * x.shape[0])
        if cfg_scale == 0.0:
            return self.inner_model.apply_model(x, t, cond=uncond)
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        cond_in = torch.cat([uncond, cond])
        eps = model.apply_model(x_in, t_in, cond=cond_in)
        eps_uncond, eps_cond = eps.chunk(2)
        return eps_uncond + (eps_cond - eps_uncond) * cfg_scale


seed = 407
n_iter = 1
batch_size = 4
seed_everything(seed)
H, W, C, f = 512, 512, 4, 8

prompt = "Photo of a fashion model with long hair, studio lightning"
reversion_prompt = "Photo of a fashion model with short hair, studio lightning"
# reversion_prompt = prompt
save_dir = "../outputs/edit_test/7"
scale = 5
inversion_scale = 5
reversion_scale = 1
plms = False
ddim_steps = 50
inversion_steps = 50
reversion_steps = 50
ddim_eta = 0.0

config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "../models/ldm/stable-diffusion-v1/model.ckpt")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
uc = model.get_learned_conditioning(batch_size * [""])
cond = model.get_learned_conditioning(batch_size * [prompt])
reversion_cond = model.get_learned_conditioning(batch_size * [reversion_prompt])
# dnw = K.external.CompVisDenoiser(model)
dnw = CFGCompVisDenoiser(model)
sigmas = dnw.get_sigmas(ddim_steps)
c_out, c_in = dnw.get_scalings(sigmas[0])
noise = torch.randn([batch_size, C, H // f, W // f], device=device)

sample_fn = lambda *args, **kwargs: K.sampling.sample_euler(*args, **kwargs, s_noise=ddim_eta)
sample_fn = K.sampling.sample_lms


# def get_eps(self, *args, **kwargs):
#     return self.inner_model.apply_model(*args, **kwargs)


def get_progress_bar(n_iter, desc=""):
    it = iter(trange(n_iter, desc=desc))
    return lambda x: next(it)


def callback(kwargs):
    i = kwargs["i"]
    x = kwargs["x"]
    c_out, c_in = dnw.get_scalings(kwargs['sigma'])
    print(f"c_out: {c_out}, c_in: {c_in}")
    print(f"step {i}: std: {(x * c_in).std()}")


def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            print(f"cond_grad * K.utils.append_dims(sigma**2, x.ndim): "
                  f"{(cond_grad * K.utils.append_dims(sigma**2, x.ndim)).abs().mean()}")
            print(f"sigma: {sigma}")
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return model_fn


def make_static_thresh_model_fn(model, value=1.):
    def model_fn(x, sigma, **kwargs):
        return model(x, sigma, **kwargs).clamp(-value, value)
    return model_fn


def l2_loss_guidance_func(original, model, grad_scale=1e-3, target="image"):
    assert target in ["image", "latent"]
    if isinstance(original, np.ndarray):
        orig_img = torch.from_numpy(original)
    original = original.to(device)
    if original.ndim == 3:
        original = original.unsqueeze(0)
    # if orig_img.dtype == torch.uint8:
    #     orig_img = orig_img.float() / 255.0
    # if orig_img.shape[-1] == 3:
    #     orig_img = orig_img.permute(0, 3, 1, 2)
    if target == 'latent':
        orig_img = model.decode_first_stage(original)
        orig_img = (orig_img + 1) / 2 * 255
        orig_img = rearrange(orig_img, "b c h w -> b h w c")

    def grad_func(x, sigma, denoised, **kwargs):
        x_0 = denoised
        if target == 'image':
            x_0 = model.differentiable_decode_first_stage(x_0)
            x_0 = (x_0 + 1) / 2 * 255
            x_0 = rearrange(x_0, "b c h w -> b h w c")
        with torch.enable_grad():
            loss = ((x_0 - original) ** 2).mean()
            grad = - torch.autograd.grad(loss, x)[0]  # / K.utils.append_dims(sigma, x.ndim)
        if target == 'latent':
            grad /= K.utils.append_dims(sigma ** 2, x.ndim)
        print(f"loss: {loss}")
        print("mean grad:", grad.abs().mean())

        if target == 'latent':
            x_0 = model.decode_first_stage(x_0)
            x_0 = (x_0 + 1) / 2 * 255
            x_0 = rearrange(x_0, "b c h w -> b h w c")

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(orig_img[1].detach().cpu().numpy().astype(np.uint8))
        axes[1].imshow(x_0[1].detach().cpu().numpy().astype(np.uint8))
        plt.show()
        return grad * grad_scale

    return grad_func

with torch.no_grad():
    with autocast('cuda'): #  autocast("cuda"):
        c_out, c_in = dnw.get_scalings(sigmas[0])
        noise = noise / c_in
        sigmas[-1] = 1e-5
        sample = sample_fn(dnw, noise, sigmas,
                           extra_args={'cond': cond, 'uncond': uc, 'cfg_scale': scale},
                           callback=get_progress_bar(ddim_steps, "sampling"))
        # c_out, c_in = dnw.get_scalings(sigmas[-1])
        # print(c_in)

        # x_samples = model.decode_first_stage(sample)
        # x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        # x_samples = x_samples * 255.0
        # x_samples = rearrange(x_samples, "b c h w -> b h w c")

        reverse_sigmas = dnw.get_sigmas(inversion_steps).flip(0)
        reverse_sigmas[0] = sigmas[-1]

        recon_noise = sample_fn(dnw, sample, reverse_sigmas,
                                extra_args={'cond': cond, 'uncond': uc, 'cfg_scale': inversion_scale},
                                callback=get_progress_bar(inversion_steps, "invert sampling"))

        # guided_model_fn = make_cond_model_fn(dnw, l2_loss_guidance_func(x_samples, model, grad_scale=1e-1))
        guided_model_fn = make_cond_model_fn(dnw, l2_loss_guidance_func(sample, model, grad_scale=1e4, target='latent'))

        # guided_model_fn = make_static_thresh_model_fn(guided_model_fn, value=1.0)

        recon_sample = sample_fn(guided_model_fn, recon_noise, sigmas,
                                 extra_args={'cond': reversion_cond, 'uncond': uc, 'cfg_scale': scale},
                                 callback=get_progress_bar(ddim_steps, "sampling"))

        c_out, c_in = dnw.get_scalings(sigmas[0])
        original_noise = recon_noise * c_in

    sample = torch.cat([sample, recon_sample])
    latent = sample.clone().detach()
    x_samples = model.decode_first_stage(sample)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_orig, x_samples_recon = x_samples.chunk(2)
    # x_samples = rearrange(x_samples, "b c h w -> b c h w")
    # x_samples_ddim = rearrange(x_samples_ddim, "b c h w -> b h w c")
    for b in range(batch_size):
        x_sample_orig = 255. * rearrange(x_samples_orig[b].cpu().numpy(), 'c h w -> h w c')
        x_sample_recon = 255. * rearrange(x_samples_recon[b].cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample_orig.astype(np.uint8))
        img_recon = Image.fromarray(x_sample_recon.astype(np.uint8))
        fig, axes = plt.subplots(1, 2)
        # axes[0].img_reconmshow(img)
        # img.save(f"{save_dir}/{prompt.replace(' ', '_')}_{i}.png")
        axes[0].imshow(x_sample_orig.astype(np.uint8))
        axes[1].imshow(x_sample_recon.astype(np.uint8))
        axes[0].set_title("Original")
        axes[1].set_title("Reconstructed")
        plt.show()