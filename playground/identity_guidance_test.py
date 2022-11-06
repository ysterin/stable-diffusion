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
import wandb


def find_large_bbox(bbox, image_shape, required_size=512):
    """Find a large bbox that contains the given bbox and has the given size.
    The large bbox should be centered on the same center as the original, unless it is too close to the edge
    of the image, in which case it should be shifted to the edge.
        Args:
        bbox: [x, y, w, h]
        image_shape: [h, w]
        required_size: int

    """
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    half_size = required_size / 2
    new_x = max(0, center_x - half_size)
    new_y = max(0, center_y - half_size)
    new_w = min(image_shape[1] - new_x, required_size)
    new_h = min(image_shape[0] - new_y, required_size)
    return new_x, new_y, new_w, new_h


def warp_matrix_to_bbox(warp_mat, image_shape):
    """finds a warp matrix that warp the image to the bbox
    Args:
        warp_mat: [3, 2]
    returns:
        bbox: [x, y, w, h]
    """
    warp_mat = warp_mat.detach().cpu().numpy()
    x = warp_mat[0, 2]
    y = warp_mat[1, 2]
    w = image_shape[1] / warp_mat[0, 0]
    h = image_shape[0] / warp_mat[1, 1]
    return x, y, w, h

def get_warp_matrix(bbox, image_shape):
    """finds a warp matrix that warp the image to the bbox
    Args:
        bbox: [x, y, w, h]
        image_shape: [h, w]
    returns:
        warp_mat: [3, 2]
    """
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    half_size = w / 2
    new_x = max(0, center_x - half_size)
    new_y = max(0, center_y - half_size)
    new_w = min(image_shape[1] - new_x, w)
    new_h = min(image_shape[0] - new_y, h)
    warp_mat = np.array([[new_w / w, 0, new_x - x],
                         [0, new_h / h, new_y - y]])
    return warp_mat


def increase_warp_matrix(warp_matrix, scale=2.0):
    warp_matrix[0, 0] *= scale
    warp_matrix[1, 1] *= scale
    return warp_matrix

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


def identity_loss_guidance_func(model, face_net, face_feats, warp_matrix=None, grad_scale=1e-3,
                                max_sigma=1.0, min_sigma=0.3):
    device = model.device
    face_feats = torch.Tensor(face_feats).to(device)
    def grad_fn(x, sigma, denoised, **kwargs):
        if sigma > max_sigma or sigma < min_sigma:
            return torch.zeros_like(x)
        device = x.device
        batch_size = x.shape[0]
        x_0 = model.differentiable_decode_first_stage(denoised)

        if warp_matrix is not None:
            warp_mat = torch.Tensor(warp_matrix)[None].to(device).repeat((batch_size, 1, 1))
            face_crop = kornia.geometry.warp_affine(x_0, warp_mat, dsize=(112, 112))
        else:
            face_crop = kornia.geometry.transform.resize(x_0, (112, 112))

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
            grad = - torch.autograd.grad(loss, x)[0]  # / K.utils_files.append_dims(sigma, x.ndim)
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

    images_dir = '../assets/sample_images/fashion_images/full body/images'
    target_image_path = os.path.join(images_dir, 'H_M.jpg')
    # target_image_path = os.path.join(images_dir, 'Fashionnova.webp')
    source_image_path = os.path.join(images_dir, 'Bobo choses.jpg')


    # images_dir = '../assets/sample_images/faces'
    # source_image_path = os.path.join(images_dir, 'Elior_portrait_1_v01.jpg')
    # target_image_path = os.path.join(images_dir, 'Shuki_portrait_2.jpg')
    # source_image = target_image
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)
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

    warp_matrix, _ = get_init_feat(target_image)
    # warp_matrix[:, :] *= 112 / H
    # target_image = cv2.warpAffine(target_image, warp_matrix, (W, H))
    # target_image = cv2.warpAffine(target_image, warp_matrix, (112, 112))

    # target_image = cv2.resize(target_image, (W, H))

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
    # warp_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    # warp_matrix = None

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
                                                   grad_scale=50, max_sigma=3.0, min_sigma=0.0)
    forward_model_fn = make_cond_model_fn(dnw, cond_fn=identity_conf_fn)
    # forward_model_fn = make_static_thresh_model_fn(forward_model_fn, 1.0)
    # forward_model_fn = dnw
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
                                 callback=get_progress_bar(ddim_steps, "sampling"))

        recon_image = model.decode_first_stage(recon_sample)
        recon_image = rearrange(recon_image, 'b c h w -> b h w c')

        recon_image = ((recon_image + 1) / 2).clamp(0, 1)
        recon_image = recon_image.cpu().numpy()
        recon_image = np.clip(recon_image * 255, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(target_image)
    axes[1].imshow(recon_image[0])
    plt.show()
    save_path = f"outputs/face_swap_test/{source_image_path.split('/')[-1].split('.')[0]}_to_{target_image_path.split('/')[-1].split('.')[0]}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(recon_image[0], cv2.COLOR_RGB2BGR))


def load_facenet(device='cuda'):
    name = 'r100'
    model_weights_path = '../insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth'
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(model_weights_path))
    return net.eval().to(device)


if __name__ == '__main__':
    main()

