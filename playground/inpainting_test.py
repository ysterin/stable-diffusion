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
import kornia
from playground.backbones import get_model
# from playground.insightface_funcs import get_init_feat, cos_loss_f
from playground.utils import CFGCompVisDenoiser, get_progress_bar
from scripts.txt2img import load_model_from_config, chunk
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from matplotlib import pyplot as plt
from img2noise import find_noise_for_image, find_noise_for_latent, pil_img_to_latent, pil_img_to_torch
from PIL import Image
import cv2
from torch import autocast
from einops import rearrange, repeat
from tqdm import trange
import os
import wandb


def blended_diffusion_model_func(model):
    def model_fn(x, sigma, **kwargs):
        denoised = model(x, sigma, **kwargs)
        if "mask" not in kwargs or "source" not in kwargs:
            return denoised
        mask = kwargs["mask"]
        source = kwargs["source"]
        # source = source * (1 - mask)
        # c_out, c_in = model.get_scalings(sigma)
        # noised_source = source + torch.randn_like(source) * K.utils.append_dims(sigma, source.ndim)
        # print("\n")
        # print(f"x: {x.std().item()}")
        # print(f"noised_source: {noised_source.std().item()}")
        # print("noised_source_1: ", (noised_source / c_in).std().item())
        # print("noised_source_2: ", (noised_source * c_in).std().item())
        # # print(f"source: {source.std().item()}")
        # print(f"denoised: {denoised.std().item()}")
        # noised_source = noised_source * c_in
        # return denoised
        return denoised * mask + source * (1 - mask)

    return model_fn


def make_mcg_guidance_fn(alpha=1.0, invert_mask=True):
    """Guidance function for Manifold Constraint Gradient (MCG)
    Described in https://arxiv.org/abs/2206.00941
    Args:
        alpha (float): weight of the guidance gradient
    """

    def mcg_guidance_func(x, sigma, denoised, **kwargs):
        if "mask" not in kwargs or "source" not in kwargs:
            return torch.zeros_like(x)
        sigma = K.utils.append_dims(sigma, x.ndim)
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        # c_in = K.utils.append_dims(c_in, x.ndim)
        mask = kwargs["mask"]
        if invert_mask:
            mask = 1 - mask
        y_0 = kwargs["source"]
        masked_y0 = y_0 * mask
        masked_denoised = denoised * mask
        mcg = ((y_0 - masked_denoised) * mask).pow(2).sum()
        grad = torch.autograd.grad(mcg, x)[0]
        print(f"mcg: {mcg.item()}")
        print(f"grad: {grad.abs().mean().item()}")
        print(f"c_in: {c_in.mean().item()}")
        print(f"sigma: {sigma.mean().item()}")
        return - (1 - mask) * (grad * alpha) / (sigma * c_in ** 2)

    return mcg_guidance_func


def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            print(f"cond_grad * K.utils.append_dims(sigma**2, x.ndim): "
                  f"{(cond_grad * K.utils.append_dims(sigma ** 2, x.ndim)).abs().mean()}")
            # print(f"sigma: {sigma.item()}")
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma ** 2, x.ndim)
        return cond_denoised

    return model_fn

@torch.no_grad()
def sample_euler_inpainting(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                            s_tmax=float('inf'), s_noise=1., x_0=None, mask=None, mcg_alpha=0.0):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    if x_0 is None and 'source' in extra_args:
        x_0 = extra_args['source']
    if mask is None and 'mask' in extra_args:
        mask = extra_args['mask']
    if mask is not None:
        mask = 1.0 - mask
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        # denoised = model(x, sigma_hat * s_in, **extra_args)
        # d = to_d(x, sigma_hat, denoised)
        c_out, c_in = model.get_scalings(sigma_hat)
        d = model.get_eps(x * c_in, model.sigma_to_t(sigma_hat)[None], **extra_args)
        denoised = x - d * sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        c_out_1, c_in_1 = model.get_scalings(sigmas[i + 1])
        # Euler method
        x_prev = x
        x = x + d * dt
        if mask is not None and x_0 is not None:
            if mcg_alpha > 0:
                # mcg_guidance = make_mcg_guidance_fn(mcg_alpha)
                with torch.enable_grad():
                    x_prev = x_prev.detach().requires_grad_()
                    denoised = model(x_prev, sigma_hat[None], **extra_args)
                    mcg_loss = ((x_0 - denoised * mask) * mask).pow(2).sum()
                    mcg_grad = torch.autograd.grad(mcg_loss, x_prev)[0]

                    # cond_grad = mcg_guidance(x_prev, sigma_hat, denoised=denoised, mask=mask, source=x_0).detach()
                x = x - mcg_alpha * mcg_grad * (1 - mask) / (c_in)
                print(f"cond_grad: {mcg_grad.abs().mean().item()}")
                print(f"cond_grad * mask / (c_in * c_in_1): {(mcg_alpha * mcg_grad * mask / (c_in * c_in_1)).std().item()}")

            x_noised = x_0 + torch.randn_like(x_0) * K.utils.append_dims(sigmas[i + 1], x_0.ndim)
            x = x * (1 - mask) + x_noised * mask
            # x = x * mask + x_noised * (1 - mask)

    return x


def main():
    # images_dir = '../data/inpainting_examples'
    # image_name = "bertrand-gabioud-CpuFzIsHYJ0"
    # image_name = "bench2"
    # image_name = "overture-creations-5sI6fQgYIuo"
    # source_image_path = os.path.join(images_dir, f"{image_name}.png")
    # face_mask_path = os.path.join(images_dir, f"{image_name}_mask.png")
    cropped_dir = '../assets/sample_images/fashion_images/full body/cropped'
    image_name = "abjx300603_billabong,w_mul_frt1"
    image_name = "H_M"
    mask_name = "head"
    source_image_path = os.path.join(cropped_dir, image_name + '.png')
    face_mask_path = os.path.join(cropped_dir, f"{image_name}_{mask_name}.png")
    source_image = cv2.imread(source_image_path)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
    seed = 4642326
    batch_size = 1
    ddim_steps = 50
    denoising_strength = 0.75
    seed_everything(seed)
    H, W, C, f = 512, 512, 4, 8
    plt.imshow(source_image)
    plt.show()

    # prompt = "An apartment complex in the city"
    prompt = "a baby sitting on a bench"
    # prompt = "a green bench in a park"
    prompt = "Emma Watson, studio lightning, realistic, fashion photoshoot, asos, perfect face, symmetric face"
    negative_prompt = "makeup, artistic, photoshop, painting, artstation, art, ugly, unrealistic, imaginative"
    scale_factor = max(H / source_image.shape[0], W / source_image.shape[1])
    mask = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor)
    mask = mask[:H]
    H, W = source_image.shape[:2]

    # prompt = "A forest full of trees"
    # mask = 255 - mask

    masked_image = (source_image * (1 - mask[:, :, None] / 255) + mask[:, :, None]).astype(np.uint8)


    # img_width, img_height = source_image.shape[1], source_image.shape[0]
    # source_image = source_image[:H, (img_width - W) // 2:(img_width - W) // 2 + W]
    # mask = mask[:H, (img_width - W) // 2:(img_width - W) // 2 + W]

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(source_image)
    axes[1].imshow(mask)
    axes[2].imshow(masked_image)
    plt.show()


    config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "../models/ldm/stable-diffusion-v1/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).half()

    dnw = CFGCompVisDenoiser(model)
    sigmas = dnw.get_sigmas(ddim_steps)
    sigmas[-1] = 1e-5
    c_out, c_in = dnw.get_scalings(sigmas[0])

    # sample_fn = K.sampling.sample_lms
    # sample_fn = lambda *args, e**kwargs: K.sampling.sample_euler(*args, **kwargs, s_noise=0.0)
    alpha = 1e-2
    sample_fn = lambda *args, **kwargs: sample_euler_inpainting(*args, **kwargs, s_noise=1.0, mcg_alpha=alpha)
    sample_fn_1 = lambda *args, **kwargs: K.sampling.sample_euler(*args, **kwargs, s_noise=1.0)
    model_fn = dnw
    mcg_cond_fn = make_mcg_guidance_fn(alpha=alpha)
    model_fn_1 = make_cond_model_fn(model_fn, mcg_cond_fn)
    model_fn_1 = blended_diffusion_model_func(model_fn_1)
    n_steps = int(ddim_steps * denoising_strength)

    with torch.no_grad():
        with autocast('cuda'):
            mask = torch.from_numpy(mask).to(device).float() / 255
            mask = mask.ceil()
            uc = model.get_learned_conditioning(batch_size * [""])
            cond = model.get_learned_conditioning(batch_size * [prompt])
            if negative_prompt is not None and negative_prompt != "":
                uc = model.get_learned_conditioning(batch_size * [negative_prompt])


            init_image = torch.from_numpy(np.array(source_image)).to(device).float() / 255
            init_image = init_image * 2 - 1
            init_image = rearrange(init_image, "h w c -> c h w")
            init_image = init_image.unsqueeze(0)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
            latent_mask = kornia.geometry.transform.resize(mask, (init_latent.shape[-2], init_latent.shape[-1]))
            latent_mask = latent_mask.unsqueeze(0).unsqueeze(0).ceil()
            # latent_mask = torch.concat([latent_mask, torch.ones_like(latent_mask)], dim=0)
            sigma = sigmas[-n_steps - 1]
            sigma = K.utils.append_dims(sigma, init_latent.ndim)
            c_out, c_in = dnw.get_scalings(sigma)
            print(f"sigma: {sigma}")
            noised_latent = (init_latent + torch.randn_like(init_latent) * sigma)
            random_noise = torch.randn_like(init_latent) / c_in
            # noised_latent = random_noise

            for scale in [1.0, 5.0, 7.5]:
                editing_cfg_scale = scale
                # recon_noise = sample_fn(dnw, init_latent, reverse_sigmas[:n_steps],
                #                                   extra_args={"cfg_scale": inversion_cfg_scale, "cond": inversion_cond,
                #                                               "uncond": uc},
                #                                   callback=get_progress_bar(n_steps, "invert sampling"))
                # init_latent = None
                # latent_mask = torch.ones_like(latent_mask)
                recon_sample = sample_fn(model_fn, noised_latent, sigmas[-n_steps-1:],
                                         extra_args={"cfg_scale": editing_cfg_scale, "cond": cond, "uncond": uc,
                                                     "mask": latent_mask, "source": init_latent},
                                         callback=get_progress_bar(ddim_steps, "sampling"))

                recon_sample_1 = sample_fn_1(model_fn_1, noised_latent, sigmas[-n_steps-1:],
                                             extra_args={"cfg_scale": editing_cfg_scale, "cond": cond, "uncond": uc,
                                                            "mask": latent_mask, "source": init_latent},
                                                callback=get_progress_bar(ddim_steps, "sampling"))


                recon_image = model.decode_first_stage(recon_sample)
                recon_image = rearrange(recon_image, 'b c h w -> b h w c')

                recon_image = ((recon_image + 1) / 2).clamp(0, 1)
                recon_image = recon_image.cpu().numpy()
                recon_image = np.clip(recon_image * 255, 0, 255).astype(np.uint8)

                recon_image_1 = model.decode_first_stage(recon_sample_1)
                recon_image_1 = rearrange(recon_image_1, 'b c h w -> b h w c')

                recon_image_1 = ((recon_image_1 + 1) / 2).clamp(0, 1)
                recon_image_1 = recon_image_1.cpu().numpy()
                recon_image_1 = np.clip(recon_image_1 * 255, 0, 255).astype(np.uint8)

                fig, axes = plt.subplots(1, 4, figsize=(40, 10))
                axes[0].imshow(source_image)
                axes[1].imshow(mask.detach().cpu().numpy())
                axes[2].imshow(recon_image[0])

                axes[3].imshow(recon_image_1[0])
                axes[0].set_title(f"Source")
                axes[1].set_title(f"Mask")
                axes[2].set_title(f"Editing scale: {scale}")
                axes[3].set_title(f"Editing scale: {scale}")

                # plt.title(f"prompt: {prompt}", loc="left")

                plt.show()


if __name__ == '__main__':
    main()


