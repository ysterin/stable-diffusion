import cv2
import math
# import matplotlib.pyplot as plt
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import torch
import kornia
from insightface.utils import face_align
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch import autocast

from backbones import get_model

from insightface_funcs import get_init_feat, cos_loss_f_numpy, cos_loss_f
from scripts.img2img import load_model_from_config


def get_intermediates_callback():
    intermediates_list = []
    def intermediates_callback(kwargs):
        intermediates_list.append(kwargs)
    return intermediates_list, intermediates_callback

def main():
    from utils import CFGCompVisDenoiser
    import k_diffusion as K

    name = 'r100'
    # model_weights_path = '/home/galgozes/insightface/model_zoo/ms1mv3_arcface_r50_fp16/backbone.pth'
    model_weights_path = '../insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth'
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(model_weights_path))
    net.eval()

    batch_size = 1
    ddim_steps = 50
    n_steps = 30

    H, W, C, f = 768, 768, 4, 8

    # '/home/galgozes/data/diffusion/CelebAMask-HQ/CelebA-HQ-img/1.jpg'
    path1 = '../assets/sample_images/faces/Shuki_portrait_2.jpg'
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (W, H))
    path2 = '../assets/sample_images/faces/Elior_portrait_1_v01.jpg'
    # path2 = '/home/galgozes/data/diffusion/CelebAMask-HQ/CelebA-HQ-img/4.jpg'
    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (W, H))
    # img2 = img1[:, ::-1, :]
    m1, img_feats1 = get_init_feat(img1)
    # m2, img_feats2 = get_init_feat(img2)


    # cos_loss = cos_loss_f_numpy(img_feats1[0], img_feats2[0])
    # tau = 1
    # constractive_loss = -np.log((np.exp(cos_loss / tau) / np.sum(np.exp(cos_loss / tau))))

    # print(cos_loss)

    config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "../models/ldm/stable-diffusion-v1/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).half()

    net = net.to(device)
    img_feats1 = torch.from_numpy(img_feats1).to(device)

    dnw = CFGCompVisDenoiser(model).half()
    sigmas = dnw.get_sigmas(ddim_steps)
    reverse_sigmas = sigmas.flip(0)
    reverse_sigmas[0] = 1e-5
    sample_fn = K.sampling.sample_lms

    def callback(kwargs):
        i = kwargs["i"]
        x = kwargs["x"]
        c_out, c_in = dnw.get_scalings(kwargs['sigma'])
        print(f"c_out: {c_out}, c_in: {c_in}")
        print(f"step {i}: std: {(x * c_in).std()}")

    img1_numpy = img1.copy()
    with torch.no_grad():
        with autocast('cuda'):
            warp_mat = torch.Tensor(m1)[None].to(device).repeat((batch_size, 1, 1))

            intermediates_list, intermediates_callback = get_intermediates_callback()
            img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255
            img1 = img1 * 2 - 1
            img1 = img1.to(device)
            latent1 = model.get_first_stage_encoding(model.encode_first_stage(img1))
            latent1 = latent1.repeat(batch_size, 1, 1, 1)
            noised1 = sample_fn(dnw, latent1, reverse_sigmas[:n_steps], extra_args={"cfg_scale": 0.0},
                                callback=intermediates_callback)

            for i, intermediates in enumerate(intermediates_list):
                sigma = intermediates['sigma']
                denoised = intermediates['denoised']
                reconstructed_img = model.decode_first_stage(denoised)
                face_crop = kornia.geometry.warp_affine(reconstructed_img, warp_mat, dsize=(112, 112))
                face_feats = net(face_crop)

                cos_loss = cos_loss_f(face_feats, img_feats1)
                print(f"face loss at step {intermediates['i']}, sigma {sigma}: {cos_loss}")

            sigma = torch.ones((1)).to(device) * sigmas[-n_steps]
            denoised1 = dnw.forward(noised1, sigma, cfg_scale=0.0)

            reconstructed_img1 = model.decode_first_stage(denoised1)

            warp_mat = torch.Tensor(m1)[None].to(device).repeat((batch_size, 1, 1))

            face_crop = kornia.geometry.warp_affine(reconstructed_img1, warp_mat, dsize=(112, 112))

            face_feats = net(face_crop)

            cos_loss = cos_loss_f(face_feats, img_feats1)
            print(cos_loss)

            reconstructed_img1 = reconstructed_img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
            reconstructed_img1 = np.clip((reconstructed_img1 + 1) / 2, 0, 1)
            reconstructed_img1 = (reconstructed_img1 * 255).astype(np.uint8)

            face_crop = face_crop.detach().cpu().numpy().transpose(0, 2, 3, 1)
            face_crop = np.clip((face_crop + 1) / 2, 0, 1)
            face_crop = (face_crop * 255).astype(np.uint8)



    fig, axes = plt.subplots(1, batch_size + 1, figsize=(5 * (batch_size + 1), 5))
    axes[0].imshow(img1_numpy)
    axes[0].set_title("Original")
    for i in range(batch_size):
        axes[i + 1].imshow(reconstructed_img1[i])
        axes[i + 1].set_title(f"Reconstructed {i}")
    plt.show()

    fig, axes = plt.subplots(1, batch_size + 1, figsize=(5 * (batch_size + 1), 5))
    axes[0].imshow(img1_numpy)
    axes[0].set_title("Original")
    for i in range(batch_size):
        axes[i + 1].imshow(face_crop[i])
        axes[i + 1].set_title(f"Face crop {i}")

    plt.show()

    # recon_sample = sample_fn(dnw, recon_noise, sigmas[-n_steps:],
    #                          extra_args={'cond': reversion_cond, 'uncond': uc, 'cfg_scale': scale},
    #                          callback=get_progress_bar(50, "sampling"))



if __name__ == "__main__":
    main()
