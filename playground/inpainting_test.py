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

def dilate_mask(mask, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = (mask > 0).astype(np.uint8)
    return mask

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

def crop_to_shape(image, height, width):
    img_height, img_width = image.shape[:2]
    if img_height / height > img_width / width:
        img_height = int(height * img_width / width)
        image = image[:img_height, ...]
        scale_factor = height / img_height
    else:
        new_width = int(width * img_height / height)
        image = image[:, img_width // 2 - new_width // 2:img_width // 2 + new_width // 2]
        img_width = new_width
        scale_factor = width / img_width
    aspect_ratio = img_width / img_height
    assert np.allclose(aspect_ratio, width / height, rtol=0.02), f"Aspect ratio mismatch: {aspect_ratio} != {width / height}, " \
                                           f"img_width: {img_width}, img_height: {img_height}"
    image = cv2.resize(image, dsize=(width, height), interpolation= cv2.INTER_LINEAR)
    print(image.shape)
    return image


def crop_center_top(image, height, width, size_scale=1.0):
    img_height, img_width = image.shape[:2]
    image = image[:height, img_width // 2 - width // 2:img_width // 2 + width // 2]
    if size_scale != 1.0:
        image = cv2.resize(image, dsize=None, fx=size_scale, fy=size_scale, interpolation=cv2.INTER_LINEAR)
    return image


def paste_center_top(image, crop, size_scale=1.0):
    image = image.copy()
    if size_scale != 1.0:
        crop = cv2.resize(crop, dsize=None, fx=1/size_scale, fy=1/size_scale, interpolation=cv2.INTER_LINEAR)
    height, width = crop.shape[:2]
    img_height, img_width = image.shape[:2]
    image[:height, img_width // 2 - width // 2:img_width // 2 + width // 2] = crop
    return image


def main():
    seed = 4642326
    batch_size = 1
    ddim_steps = 100
    denoising_strength = 0.35
    scale = 15

    seed_everything(seed)
    # H, W, C, f = 960, 640, 4, 8
    # face_crop_size = 256
    # face_crop_scale = 2.0

    H, W, C, f = 1280 + 64 + 64, 960, 4, 8
    face_crop_size = 512
    face_crop_scale = 2.0

    # prompt = "An apartment complex in the city"
    prompt = "a baby sitting on a bench"
    # prompt = "a green bench in a park"
    prompt = "Emma Watson, studio lightning, realistic, fashion photoshoot, asos, perfect face, symmetric face"
    prompt = "shukistern guy"
    # prompt = "Barack Obama"
    negative_prompt = "makeup, artistic, photoshop, painting, artstation, art, ugly, unrealistic, imaginative"

    config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "../models/ldm/stable-diffusion-v1/model_shuki1.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    cropped_dir = '../assets/sample_images/fashion_images/full body/uncropped'
    outputs_dir = f"outputs/inpainting/fashion_images/prompts/{prompt.replace(' ', '_')}"
    os.makedirs(outputs_dir, exist_ok=True)
    mask_name = "body"
    mask_dilation = 8
    image_list = ["Terminal_tank_1", "Terminal_tank_2", "Terminal_tank_3"]

    for image_name in image_list:
        source_image_path = os.path.join(cropped_dir, image_name + '.png')
        body_mask_path = os.path.join(cropped_dir, f"{image_name}_body.png")
        head_mask_path = os.path.join(cropped_dir, f"{image_name}_head.png")
        source_image = cv2.imread(source_image_path)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(body_mask_path, cv2.IMREAD_GRAYSCALE)
        head_mask = cv2.imread(head_mask_path, cv2.IMREAD_GRAYSCALE)

        plt.imshow(source_image)
        plt.show()

        source_image = crop_to_shape(source_image, H, W)
        mask = crop_to_shape(mask, H, W)
        head_mask = crop_to_shape(head_mask, H, W)

        mask = dilate_mask(mask, kernel_size=mask_dilation)
        head_mask = dilate_mask(head_mask, kernel_size=mask_dilation)

        face_crop_mask_1 = crop_center_top(head_mask, face_crop_size, face_crop_size, size_scale=1.0)
        face_crop_mask_2 = crop_center_top(head_mask, face_crop_size, face_crop_size, size_scale=2.0)

        for scale in [10, 15, 20][1:]:
            for denoising_strength in [0.2, 0.3, 0.35, 0.4, 0.5][2:]:
                _, recon_image = inpaint_image(source_image, mask, model, prompt, negative_prompt,
                                                  ddim_steps=ddim_steps, denoising_strength=denoising_strength,
                                                  cfg_scale=scale, device=device)

                face_crop_1 = crop_center_top(recon_image[0].copy(), face_crop_size, face_crop_size, size_scale=1.0).copy()
                face_crop_2 = crop_center_top(recon_image[0].copy(), face_crop_size, face_crop_size, size_scale=2.0).copy()

                _, recon_face_crop_1 = inpaint_image(face_crop_1, face_crop_mask_1, model, prompt, negative_prompt,
                                                    ddim_steps=ddim_steps, denoising_strength=denoising_strength,
                                                    cfg_scale=scale, device=device)

                _, recon_face_crop_2 = inpaint_image(face_crop_2, face_crop_mask_2, model, prompt, negative_prompt,
                                                    ddim_steps=ddim_steps, denoising_strength=denoising_strength,
                                                    cfg_scale=scale, device=device)


                # combined_image = (0.5 * source_image + 0.5 * recon_image[0]).astype(np.uint8)
                w = 3
                # recon_face_crop[:, w] = 0
                # recon_face_crop[:, -w:] = 0
                # recon_face_crop[:, :, :w] = 0
                # recon_face_crop[:, :, -w:] = 0
                recon_face_crop_1[:, w] = 0
                recon_face_crop_1[:, -w:] = 0
                recon_face_crop_1[:, :, :w] = 0
                recon_face_crop_1[:, :, -w:] = 0
                recon_face_crop_2[:, w] = 0
                recon_face_crop_2[:, -w:] = 0
                recon_face_crop_2[:, :, :w] = 0
                recon_face_crop_2[:, :, -w:] = 0


                recon_image_pasted_1 = paste_center_top(recon_image[0], recon_face_crop_1[0], size_scale=1.0)
                recon_image_pasted_2 = paste_center_top(recon_image[0], recon_face_crop_2[0], size_scale=2.0)

                # fig, axes = plt.subplots(1, 5, figsize=(60, 15))
                # axes[0].imshow(source_image)
                # axes[1].imshow(mask)
                # axes[2].imshow(recon_image[0])
                # axes[3].imshow(recon_image_pasted)
                # axes[4].imshow(combined_image)
                #
                # # axes[3].imshow(recon_image_1[0])
                # axes[0].set_title(f"Source")
                # axes[1].set_title(f"Mask")
                # axes[2].set_title(f"Editing scale: {scale}")
                # axes[3].set_title(f"Editing scale: {scale}")

                # plt.title(f"prompt: {prompt}", loc="left")

                plt.show()
                rec_img = cv2.cvtColor(recon_image[0], cv2.COLOR_RGB2BGR)
                rec_img_pasted_1 = cv2.cvtColor(recon_image_pasted_1, cv2.COLOR_RGB2BGR)
                rec_img_pasted_2 = cv2.cvtColor(recon_image_pasted_2, cv2.COLOR_RGB2BGR)
                # rec_img_1 = cv2.cvtColor(recon_image_1[0], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{denoising_strength}_{scale}_{ddim_steps}_{H}_{W}.png"), rec_img)
                cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{denoising_strength}_{scale}_{ddim_steps}_{H}_{W}_pasted_1.png"), rec_img_pasted_1)
                cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{denoising_strength}_{scale}_{ddim_steps}_{H}_{W}_pasted_2.png"), rec_img_pasted_2)

                # cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{denoising_strength}_{scale}_{ddim_steps}_{H}_{W}_combined.png"), combined_img)
                # cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{denoising_strength}_{scale}_1.png"), rec_img_1)


def inpaint_image(source_image, mask, model, prompt, negative_prompt, ddim_steps, denoising_strength=0.5, cfg_scale=10,
                  device='cuda', batch_size=1, alpha=0, batch_cond_uncond=False):
    # masked_image = (source_image * (1 - mask[:, :, None] / 255) + mask[:, :, None]).astype(np.uint8)
    n_steps = int(denoising_strength * ddim_steps)
    # fig, axes = plt.subplots(1, 3)
    # axes[0].imshow(source_image)
    # axes[1].imshow(mask)
    # axes[2].imshow(masked_image)
    # plt.show()
    dnw = CFGCompVisDenoiser(model.to(device), batch_cond_uncond=batch_cond_uncond, device=device)
    sigmas = dnw.get_sigmas(ddim_steps).to(device)
    sigmas[-1] = 1e-5
    sample_fn = lambda *args, **kwargs: sample_euler_inpainting(*args, **kwargs, s_noise=1.0, mcg_alpha=alpha)
    model_fn = dnw
    # model.to('cpu')
    with torch.no_grad():
        with autocast('cuda', dtype=torch.float16):
        # with nullcontext():
        #     model.cond_stage_model.to(device)
            mask = torch.from_numpy(mask).to(device).float() / 255
            mask = mask.ceil()
            uc = model.get_learned_conditioning(batch_size * [""])
            cond = model.get_learned_conditioning(batch_size * [prompt])
            if negative_prompt is not None and negative_prompt != "":
                uc = model.get_learned_conditioning(batch_size * [negative_prompt])

            # model.cond_stage_model.to('cpu')

            init_image = torch.from_numpy(np.array(source_image)).to(device).float() / 255
            init_image = init_image * 2 - 1
            init_image = rearrange(init_image, "h w c -> c h w")
            init_image = init_image.unsqueeze(0)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

            # model.first_stage_model.to(device)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
            # model.first_stage_model.to('cpu')
            latent_mask = kornia.geometry.transform.resize(mask, (init_latent.shape[-2], init_latent.shape[-1]))
            latent_mask = latent_mask.unsqueeze(0).unsqueeze(0).ceil()
            # latent_mask = torch.concat([latent_mask, torch.ones_like(latent_mask)], dim=0)
            sigma = sigmas[-n_steps - 1]
            sigma = K.utils.append_dims(sigma, init_latent.ndim).to(device)
            # c_out, c_in = dnw.get_scalings(sigma)
            noised_latent = (init_latent + torch.randn_like(init_latent) * sigma)
            # random_noise = torch.randn_like(init_latent) / c_in

            editing_cfg_scale = cfg_scale
            # model.model.to(device)
            recon_sample = sample_fn(model_fn, noised_latent, sigmas[-n_steps - 1:],
                                     extra_args={"cfg_scale": editing_cfg_scale, "cond": cond, "uncond": uc,
                                                 "mask": latent_mask, "source": init_latent},
                                     callback=get_progress_bar(ddim_steps, "sampling"))
            # model.model.to('cpu')
            # model.first_stage_model.to(device)
            recon_image = model.decode_first_stage(recon_sample)
            # model.first_stage_model.to('cpu')
            recon_image = rearrange(recon_image, 'b c h w -> b h w c')

            recon_image = ((recon_image + 1) / 2).clamp(0, 1)
            recon_image = recon_image.cpu().numpy()
            recon_image = np.clip(recon_image * 255, 0, 255).astype(np.uint8)

    return mask, recon_image


@contextmanager
def timethis(name):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{name} took {end - start} seconds")


if __name__ == '__main__':
    with timethis("main"):
        main()


