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
        # c_out, c_in = model.get_scalings(sigma)
        noised_source = source + torch.randn_like(source) * K.utils.append_dims(sigma, source.ndim)
        # print("c_out: ", c_out)
        # print("c_in: ", c_in)
        # print("sigma: ", sigma)
        # print(f"denoised std: {denoised.std().item():.3f}")
        # print(f"noised_source std: {noised_source.std().item():.3f}")
        # noised_source_cin = noised_source / c_in
        # print(f"noised_source * c_in std: {noised_source_cin.std().item():.3f}")
        return denoised * mask + source * (1 - mask)

    return model_fn


def make_mcg_guidance_fn(alpha=1.0):
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
        mask = 1.0 - kwargs["mask"]
        y_0 = kwargs["source"]
        masked_y0 = y_0 * mask
        masked_denoised = denoised * mask
        mcg = (masked_y0 - masked_denoised).pow(2).sum()
        grad = torch.autograd.grad(mcg, x)[0]
        print(f"mcg: {mcg.item()}")
        print(f"grad: {grad.abs().mean().item()}")
        print(f"c_in: {c_in.mean().item()}")
        print(f"sigma: {sigma.mean().item()}")
        return - (grad * alpha) / (sigma ** 2 * c_in ** 2)

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


def main():
    images_dir = '../assets/sample_images/fashion_images/full body/images'
    parsing_dir = '../assets/sample_images/fashion_images/full body/vis_human_parsing'
    source_image_path = os.path.join(images_dir, 'H_M.jpg')
    face_mask_path = os.path.join(parsing_dir, 'H_M_face.png')

    # images_dir = '../assets/sample_images/faces'
    # source_image_path = os.path.join(images_dir, 'Elior_portrait_1_v01.jpg')
    # target_image_path = os.path.join(images_dir, 'Shuki_portrait_2.jpg')
    # source_image = target_image
    source_image = cv2.imread(source_image_path)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
    seed = 407
    batch_size = 1
    ddim_steps = 50
    seed_everything(seed)
    H, W, C, f = 512, 512, 4, 8
    plt.imshow(source_image)
    plt.show()
    name = "Emma Watson"
    inversion_prompt = "Professional full body fashion photoshoot of a female fashion model"
    editing_prompt = f"Professional full body fashion photoshoot of {name} as fashion model"
    # editing_prompt = f"Professional full body fashion photoshoot of a female fashion model with the face of {name}"
    suffix = ", by gregsdiary oxana gromova, full body shot, professional fashion photoshoot, very detailed, xf iq4, 50mp, 50mm, f/1.4, iso 200, 1/160s, 8K"
    suffix = ""
    inversion_prompt = f"Female fashion model{suffix}"
    editing_prompt = f"Full body portrait of {name}{suffix}"
    editing_prompt = f"{name}"
    # editing_prompt = f"Female fashion model with dark skin, {suffix}"
    editing_prompt = inversion_prompt
    # editing_prompt = ""
    # scale_factor = max(H / source_image.shape[0], W / source_image.shape[1])
    # source_image = cv2.resize(source_image, (0, 0), fx=scale_factor, fy=scale_factor)
    # source_image = source_image[:H]
    # mask = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor)
    # mask = mask[:H]
    # H, W = source_image.shape[:2]

    img_width, img_height = source_image.shape[1], source_image.shape[0]
    source_image = source_image[:H, (img_width - W) // 2:(img_width - W) // 2 + W]
    mask = mask[:H, (img_width - W) // 2:(img_width - W) // 2 + W]

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(source_image)
    axes[1].imshow(mask)
    plt.show()

    config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "../models/ldm/stable-diffusion-v1/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).half()

    dnw = CFGCompVisDenoiser(model)
    sigmas = dnw.get_sigmas(ddim_steps)
    sigmas[-1] = 1e-5
    c_out, c_in = dnw.get_scalings(sigmas[0])

    sample_fn = K.sampling.sample_lms
    inversion_sample_fn = lambda *args, **kwargs: K.sampling.sample_euler(*args, **kwargs, s_noise=0)
    forward_sample_fn = lambda *args, **kwargs: K.sampling.sample_euler(*args, **kwargs, s_noise=0.0)
    inversion_cfg_scale = 0.0
    editing_cfg_scale = 2.0
    model_fn = dnw
    mcg_cond_fn = make_mcg_guidance_fn(alpha=1e-1)
    model_fn = make_cond_model_fn(model_fn, mcg_cond_fn)
    editing_model_fn = blended_diffusion_model_func(model_fn)
    n_steps = int(ddim_steps * 1.0)
    with torch.no_grad():
        with autocast('cuda'):
            mask = torch.from_numpy(mask).to(device).float() / 255
            uc = model.get_learned_conditioning(batch_size * [""])
            editing_cond = model.get_learned_conditioning(batch_size * [editing_prompt])
            inversion_cond = model.get_learned_conditioning(batch_size * [inversion_prompt])

            init_image = torch.from_numpy(np.array(source_image)).to(device).float() / 255
            init_image = init_image * 2 - 1
            init_image = rearrange(init_image, "h w c -> c h w")
            init_image = init_image.unsqueeze(0)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

            # init_latent_dist = model.vae.encode(init_image).latent_dist
            # init_latents = init_latent_dist.sample(generator=None)
            #
            # init_latent = 0.18215 * init_latents

            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
            latent_mask = kornia.geometry.transform.resize(mask, (init_latent.shape[-2], init_latent.shape[-1]))
            latent_mask = latent_mask.unsqueeze(0).unsqueeze(0)
            latent_mask = torch.concat([latent_mask, torch.ones_like(latent_mask)], dim=0)
            reverse_sigmas = dnw.get_sigmas(ddim_steps).flip(0)
            reverse_sigmas[0] = 1e-5

            for scale in [0.0, 1.0, 2.0, 4.0, 5.0, 7.5, 10.0, 12.5]:
                editing_cfg_scale = scale
                recon_noise = inversion_sample_fn(dnw, init_latent, reverse_sigmas[:n_steps],
                                                  extra_args={"cfg_scale": inversion_cfg_scale, "cond": inversion_cond,
                                                              "uncond": uc},
                                                  callback=get_progress_bar(n_steps, "invert sampling"))

                # recon_sample = sample_fn(editing_model_fn, recon_noise, sigmas,
                #                          extra_args={"cfg_scale": editing_cfg_scale, "cond": editing_cond, "uncond": uc,
                #                                      "mask": latent_mask, "source": torch.cat([init_latent, init_latent], dim=0)},
                #                          callback=get_progress_bar(ddim_steps, "sampling"))

                recon_noise = torch.cat([recon_noise, recon_noise], dim=0)
                recon_sample = forward_sample_fn(editing_model_fn, recon_noise, sigmas[-n_steps:],
                                                 extra_args={"cfg_scale": editing_cfg_scale,
                                                             "cond": torch.cat([editing_cond, editing_cond], dim=0),
                                                             "uncond": torch.cat([uc, uc], dim=0),
                                                             "mask": latent_mask,
                                                             "source": torch.cat([init_latent, init_latent], dim=0)},
                                                 callback=get_progress_bar(n_steps, "sampling"))

                c_out, c_in = dnw.get_scalings(sigmas[0])
                random_noise = torch.randn_like(recon_noise[:]) / c_in
                random_sample = forward_sample_fn(editing_model_fn, random_noise, sigmas,
                                                  # extra_args={"cfg_scale": editing_cfg_scale, "cond": editing_cond,
                                                  #             "uncond": uc},
                                                  extra_args={"cfg_scale": editing_cfg_scale,
                                                              "cond": torch.cat([editing_cond, editing_cond], dim=0),
                                                              "uncond": torch.cat([uc, uc], dim=0),
                                                              "mask": latent_mask,
                                                              "source": torch.cat([init_latent, init_latent], dim=0)},

                                                  callback=get_progress_bar(ddim_steps, "sampling"))

                recon_image = model.decode_first_stage(recon_sample)
                recon_image = rearrange(recon_image, 'b c h w -> b h w c')

                recon_image = ((recon_image + 1) / 2).clamp(0, 1)
                recon_image = recon_image.cpu().numpy()
                recon_image = np.clip(recon_image * 255, 0, 255).astype(np.uint8)

                random_image = model.decode_first_stage(random_sample)
                random_image = rearrange(random_image, 'b c h w -> b h w c')

                random_image = ((random_image + 1) / 2).clamp(0, 1)
                random_image = random_image.cpu().numpy()
                random_image = np.clip(random_image * 255, 0, 255).astype(np.uint8)

                fig, axes = plt.subplots(1, 5, figsize=(40, 10))
                axes[0].imshow(source_image)
                axes[1].imshow(recon_image[0])
                axes[2].imshow(recon_image[1])
                axes[3].imshow(random_image[0])
                axes[4].imshow(random_image[1])
                axes[0].set_title("Source")
                axes[1].set_title(f"Editing with mask, scale={editing_cfg_scale}")
                axes[2].set_title(f"Inversion without mask, scale={editing_cfg_scale}")
                axes[3].set_title(f"Random with mask (inpainting), scale={editing_cfg_scale}")
                axes[4].set_title(f"Random without mask, scale={editing_cfg_scale}")
                plt.show()


def load_facenet(device='cuda'):
    name = 'r100'
    model_weights_path = '../insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth'
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(model_weights_path))
    return net.eval().to(device)


if __name__ == '__main__':
    main()
