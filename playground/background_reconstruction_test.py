import glob
import json

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
# from playground.backbones import get_model
# from playground.insightface_funcs import get_init_feat, cos_loss_f
from playground.utils import CFGCompVisDenoiser, get_progress_bar
from playground.utils_files.source_crops import get_3ddfa_face_crop
from scripts.txt2img import load_model_from_config, chunk
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from matplotlib import pyplot as plt
from img2noise import find_noise_for_image, find_noise_for_latent, pil_img_to_latent, pil_img_to_torch
from PIL import Image
import cv2
import copy
from torch import autocast
from einops import rearrange, repeat
from tqdm import trange
import os
from inpainting_test import sample_euler_inpainting, background_reconstruction, get_best_human_ind, warp_im_alphapose, \
    get_crop_by_head, get_warp_im, dilate_mask, inpaint_image, get_alpha_and_face


def main():
    seed = 4642326
    ddim_steps = 100

    seed_everything(seed)
    # face_crop_size = 256
    # face_crop_scale = 2.0
    # H, W, C, f = 960, 640, 4, 8
    H, W, C, f = 1280 + 64 + 64, 768, 4, 8
    # H, W, C, f = 960, 640, 4, 8
    # H, W, C, f = 1536, 832, 4, 8
    # H, W, C, f = 768, 512, 4, 8
    face_crop_size = 512
    face_crop_scale = 1.5
    use_head = True
    use_face = True
    use_init_head = False
    body_denoising_strength = 0.5
    head_denoising_strength = 0.65
    face_denoising_strength = 0.5  # 0.35
    bodyhead_mcg_alpha = -1e-1
    bodyheadface_mcg_alpha = -1e-1
    scale = 10
    # prompt = "An apartment complex in the city"
    # prompt = "a baby sitting on a bench"
    # prompt = "a green bench in a park"
    # prompt = "Emma Watson, studio lightning, realistic, fashion photoshoot, asos, perfect face, symmetric face"
    prompt = "qwv man, german, european, caucasian, bright skin"
    negative_prompt = "makeup, artistic, photoshop, painting, artstation, art, ugly, unrealistic, imaginative, african, blurry"

    config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
    model_name = "avi_dual_face_3ddfa_qwv"
    # model = load_model_from_config(config, f"../../dreambooth_sd/stable_diffusion_weights/{model_name}/model.ckpt")
    model = load_model_from_config(config, f"../models/ldm/stable-diffusion-v1/dreambooth/{model_name}/model.ckpt")
    target_scales = [350, 210]  # depends on the crop. full head 210, tight face 350
    enlarge_bys = [0.2, 0.7]  # depende on the crop. full head 0.7, tight face 0.2

    image_list1 = []
    use_face1 = []
    src_dir1 = '../assets/sample_images/fashion_images/full body/'
    src_video_alpha_info1, src_info_face_pca1 = get_alpha_and_face(src_dir1)
    image_list1 = ["Terminal_tank_1", "Terminal_tank_2", "Terminal_tank_3"]
    use_face1 = [1, 1, 1]
    # image_list1 = ["Terminal_tank_3"]
    # use_face1 = [1]

    src_dir2 = '../assets/sample_images/fashion_images/target_ims/'
    src_video_alpha_info2, src_info_face_pca2 = get_alpha_and_face(src_dir2)
    image_list2 = ["terminal_cauc_back",  "terminal_cauc_full", "terminal_cauc_side",
                  "terminal_dark", "terminal_dark_back", "terminal_dark_big"]
    use_face2 = [0, 1, 1, 1, 0, 1]
    # image_list2 = ["terminal_dark", "terminal_dark_back", "terminal_dark_big"]
    # use_face2 = [1, 0, 1]
    # image_list2 = ["terminal_dark_back"]
    # use_face2 = [0]

    idxs1 = [0, 1]
    idxs2 = [0, 5]

    # image_list1 = [image_list1[i] for i in idxs1]
    # use_face1 = [use_face1[i] for i in idxs1]
    # image_list2 = [image_list2[i] for i in idxs2]
    # use_face2 = [use_face2[i] for i in idxs2]

    num_ims_dir1 = len(image_list1)


    use_faces = use_face1 + use_face2
    image_list = image_list1 + image_list2

    src_video_alpha_info, src_info_face_pca = src_video_alpha_info1, src_info_face_pca1
    src_video_alpha_info.update(src_video_alpha_info2)
    src_info_face_pca.update(src_info_face_pca2)


    outputs_dir = f"outputs/inpainting/fashion_images/prompts/{prompt.replace(' ', '_').replace(',', '')}/{model_name}_bodyheadface_detailed_prompt"
    os.makedirs(outputs_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    mask_dilation = 8

    src_dir = src_dir1
    for image_ind, image_name in enumerate(tqdm(image_list, 'target_image')):
        print(f"\n\n--------------\n*** Generating images of {image_name} ***\n")
        if image_ind == num_ims_dir1:
            src_dir = src_dir2
        use_face_cur = use_faces[image_ind]
        cropped_dir = os.path.join(src_dir, 'uncropped')
        source_image_path = os.path.join(cropped_dir, image_name + '.png')
        body_mask_path = os.path.join(cropped_dir, f"{image_name}_body.png")
        head_mask_path = os.path.join(cropped_dir, f"{image_name}_head.png")
        clothes_mask_path = os.path.join(cropped_dir, f"{image_name}_clothes.png")

        source_image = cv2.imread(source_image_path)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(body_mask_path, cv2.IMREAD_GRAYSCALE)
        head_mask = cv2.imread(head_mask_path, cv2.IMREAD_GRAYSCALE)
        clothes_mask = cv2.imread(clothes_mask_path, cv2.IMREAD_GRAYSCALE)

        image_full_name = [im_name for im_name in src_video_alpha_info.keys() if image_name+'.' in im_name][0]
        frame_alpha_data = src_video_alpha_info[image_full_name]
        frame_face_data = src_info_face_pca[image_full_name]
        best_human_ind = get_best_human_ind(frame_alpha_data)
        human_face = frame_face_data[str(best_human_ind)]
        human_alpha = frame_alpha_data['humans'][best_human_ind]

        source_image, mask, clothes_mask, warped_head_mask, warped_body_kp, warped_face_kp, warped_face_pca_head_bbox = \
            warp_im_alphapose(source_image, mask, clothes_mask, head_mask, human_alpha, human_face, H, W)

        mask = dilate_mask(mask, kernel_size=mask_dilation//2)
        mask = (1 - clothes_mask // 255) * mask  # restricted

        # body_kp = warped_body_kp
        # face_kp = warped_face_kp
        # face_pca_head_bbox = warped_face_pca_head_bbox
        #
        # target_scale_face = target_scales[0]
        # enlarge_by_face = enlarge_bys[0]
        # target_scale_head = target_scales[1]
        # enlarge_by_head = enlarge_bys[1]

        # _, recon_image = inpaint_image(source_image.copy(), mask.copy(), model, prompt, negative_prompt,
        #                                ddim_steps=ddim_steps, denoising_strength=body_denoising_strength,
        #                                cfg_scale=scale, device=device)

        reconstructed_image = reconstruction_test(source_image, model, device=device, name=image_name)


def reconstruction_test(image, model, device='cuda', name=None):
    orig_image = image.copy()
    with torch.no_grad():
        with autocast('cuda', dtype=torch.float16):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            image = (image / 255.0) * 2.0 - 1.0
            image = image.half()
            latent = model.get_first_stage_encoding(model.encode_first_stage(image))
            decoded = model.decode_first_stage(latent)
            mask = torch.zeros_like(image[:, 0]).to(device)
            latent_mask = torch.zeros_like(latent[:, 0]).to(device)
            reconstructed = background_reconstruction(latent, latent, image, mask, latent_mask, model, n_steps=1000,
                                                      lr=5e-3, gamma=0.9, n_stages=20, name=name)

            decoded = touint8(decoded)
            reconstructed = touint8(reconstructed)

    fig, axes = plt.subplots(1, 3, figsize=(45, 15))
    axes[0].imshow(orig_image)
    axes[1].imshow(reconstructed)
    axes[2].imshow(decoded)

    plt.show()
    return decoded


def touint8(batch):
    batch = batch.float()
    batch = (batch + 1.0) / 2.0
    batch = batch.clamp(0, 1) * 255.0
    batch = batch.permute(0, 2, 3, 1).cpu().numpy()
    batch = batch.astype(np.uint8)
    batch = batch[0]
    return batch


if __name__ == '__main__':
    main()