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
from playground.insightface_funcs import get_init_feat, cos_loss_f
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

def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            print(f"cond_grad * K.utils.append_dims(sigma**2, x.ndim): "
                  f"{(cond_grad * K.utils.append_dims(sigma**2, x.ndim)).abs().mean()}")
            print(f"sigma: {sigma.item()}")
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return model_fn

def make_static_thresh_model_fn(model, value=1.):
    def model_fn(x, sigma, **kwargs):
        return model(x, sigma, **kwargs).clamp(-value, value)
    return model_fn


def identity_loss_guidance_func(model, face_net, face_feats, warp_matrix, grad_scale=1e-3, max_sigma=1.0):
    device = model.device
    face_feats = torch.Tensor(face_feats).to(device)
    def grad_fn(x, sigma, denoised, **kwargs):
        if sigma > max_sigma:
            return torch.zeros_like(x)
        device = x.device
        batch_size = x.shape[0]
        x_0 = model.differentiable_decode_first_stage(denoised)
        warp_mat = torch.Tensor(warp_matrix)[None].to(device).repeat((batch_size, 1, 1))

        face_crop = kornia.geometry.warp_affine(x_0, warp_mat, dsize=(112, 112))

        predicted_face_feats = face_net(face_crop)
        cos_loss = cos_loss_f(face_feats, predicted_face_feats)
        print(f"cos_loss: {cos_loss.item()}")
        grad = torch.autograd.grad(cos_loss, x)[0]
        print("mean grad:", grad.abs().mean())
        face_crop = face_crop.detach().cpu().numpy().transpose(0, 2, 3, 1)
        face_crop = np.clip((face_crop + 1) / 2, 0, 1)
        face_crop = (face_crop * 255).astype(np.uint8)
        plt.imshow(face_crop[0])
        plt.show()

        return grad * grad_scale
    return grad_fn


def l2_loss_guidance_func(original, model, grad_scale=1e-3, target="image", device='cuda'):
    assert target in ["image", "latent"]
    if isinstance(original, np.ndarray):
        orig_img = torch.from_numpy(original)
    original = original.to(device)
    if original.ndim == 3:
        original = original.unsqueeze(0)
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



def main():

    # images_dir = '../assets/sample_images/fashion_images/full body/images'
    # source_image = os.path.join(images_dir, 'Fashionnova.webp')
    # target_image = os.path.join(images_dir, 'H_M.jpg')
    images_dir = '../assets/sample_images/faces'
    source_image = os.path.join(images_dir, 'Elior_portrait_1_v01.jpg')
    target_image = os.path.join(images_dir, 'Shuki_portrait_2.jpg')
    # source_image = target_image
    source_image = cv2.imread(source_image)
    target_image = cv2.imread(target_image)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    seed = 407
    n_iter = 1
    batch_size = 1
    ddim_steps = 50
    seed_everything(seed)
    H, W, C, f = 512, 512, 4, 8
    plt.imshow(target_image)
    plt.show()

    scale_factor = max(H / target_image.shape[0], W / target_image.shape[1])
    target_image = cv2.resize(target_image, (0, 0), fx=scale_factor, fy=scale_factor)
    target_image = target_image[:H]
    H, W = target_image.shape[:2]

    plt.imshow(target_image)
    plt.show()

    # source_image = np.asarray(source_image)
    # target_image = np.asarray(target_image)

    # m1, img_feats1 = get_init_feat(img1)
    _, source_face_feats = get_init_feat(source_image)
    warp_matrix, _ = get_init_feat(target_image)

    config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "../models/ldm/stable-diffusion-v1/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).half()

    dnw = CFGCompVisDenoiser(model)
    sigmas = dnw.get_sigmas(ddim_steps)
    sigmas[-1] = 1e-5
    c_out, c_in = dnw.get_scalings(sigmas[0])

    facenet = load_facenet(device=device)

    sample_fn = K.sampling.sample_lms
    identity_conf_fn = identity_loss_guidance_func(model, facenet, source_face_feats, warp_matrix,
                                                   grad_scale=100, max_sigma=5.0)
    forward_model_fn = make_cond_model_fn(dnw, cond_fn=identity_conf_fn)
    # forward_model_fn = make_static_thresh_model_fn(forward_model_fn, 1.0)

    with autocast('cuda'):
        init_image = torch.from_numpy(np.array(target_image)).to(device).float() / 255
        init_image = init_image * 2 - 1
        init_image = rearrange(init_image, "h w c -> c h w")
        init_image = init_image.unsqueeze(0)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        reverse_sigmas = dnw.get_sigmas(ddim_steps).flip(0)
        reverse_sigmas[0] = 1e-5

        recon_noise = sample_fn(dnw, init_latent, reverse_sigmas, extra_args={"cfg_scale": 0},
                                callback=get_progress_bar(ddim_steps, "invert sampling"))

        recon_sample = sample_fn(forward_model_fn, recon_noise, sigmas, extra_args={"cfg_scale": 0},
                                 callback=get_progress_bar(50, "sampling"))

        recon_image = model.decode_first_stage(recon_sample)
        recon_image = rearrange(recon_image, 'b c h w -> b h w c')

        recon_image = ((recon_image + 1) / 2).clamp(0, 1)
        recon_image = recon_image.cpu().numpy()
        recon_image = np.clip(recon_image * 255, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(target_image)
    axes[1].imshow(recon_image[0])
    plt.show()


def load_facenet(device='cuda'):
    name = 'r100'
    model_weights_path = '../insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth'
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(model_weights_path))
    return net.eval().to(device)


if __name__ == '__main__':
    main()

