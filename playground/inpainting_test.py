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
        # c_in = K.utils_files.append_dims(c_in, x.ndim)
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


def warp_im_alphapose(img, mask, clothes_mask, head_mask, human_alpha, human_face, H, W):
    body_kp = np.array(human_alpha['body'])
    face_kp = np.array(human_face['landmark']).T
    face_pca_head_bbox = np.array(human_face['head_bounding_box'])
    bbox = np.array(human_alpha['bbox'])

    x1, y1, x2, y2 = enlarge_bbox(bbox, img.shape[0], img.shape[1], 0.1, 0.1)
    if y2 - y1 > x2 - x1:
        # box_size = (y2 - y1) / img.shape[0]
        # box_size = 1 / box_size * (H / img.shape[0])
        box_size = H / (y2 - y1)
    else:
        # box_size = (x2 - x1) / img.shape[1]
        # box_size = 1 / box_size * (target_width / img.shape[1])
        box_size = W / (x2 - x1)

    box_size *= 1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    x_offset = (W - 1.0) / 2.0 - center_x * box_size
    y_offset = (H - 1.0) / 2.0 - center_y * box_size

    warp_matrix = np.float32([[box_size, 0, x_offset], [0, box_size, y_offset]])
    warped_im = cv2.warpAffine(img, warp_matrix, (W, H))
    warped_mask = cv2.warpAffine(mask, warp_matrix, (W, H))
    warped_head_mask = cv2.warpAffine(head_mask, warp_matrix, (W, H))
    warped_clothes_mask = cv2.warpAffine(clothes_mask, warp_matrix, (W, H))

    rot_mat3d = np.vstack((warp_matrix, [0, 0, 1]))

    def prepare_kp_for_transofrm(kp_array):
        kp = [[kp_i[:2]] for kp_i in kp_array]
        return np.array(kp)
    body_kp_no_conf = prepare_kp_for_transofrm(body_kp)
    face_kp_no_conf = prepare_kp_for_transofrm(face_kp)
    face_pca_head_bbox_forT = np.array([[face_pca_head_bbox[:2]], [face_pca_head_bbox[2:]]], dtype=float)

    warped_body_kp = cv2.perspectiveTransform(body_kp_no_conf, rot_mat3d)
    warped_body_kp = np.array([np.append(warped_body_kp[i, 0], [body_kp[i, 2]]) for i in range(len(warped_body_kp))])
    warped_face_kp = cv2.perspectiveTransform(face_kp_no_conf, rot_mat3d)
    warped_face_kp = np.array([warped_face_kp[i, 0] for i in range(len(warped_face_kp))])
    warped_face_pca_head_bbox = cv2.perspectiveTransform(face_pca_head_bbox_forT, rot_mat3d)

    return warped_im, warped_mask, warped_clothes_mask, warped_head_mask, warped_body_kp, warped_face_kp, warped_face_pca_head_bbox.flatten().astype(int)


def warp_to_shape(image, height, width):
    img_height, img_width = image.shape[:2]
    if img_height / height > img_width / width:
        img_height = int(height * img_width / width)
        # image = image[:img_height, ...]
        # scale_factor = height / img_height
    else:
        new_width = int(width * img_height / height)
        # image = image[:, img_width // 2 - new_width // 2:img_width // 2 + new_width // 2]
        img_width = new_width
        scale_factor = width / img_width
    aspect_ratio = img_width / img_height
    # assert np.allclose(aspect_ratio, width / height, rtol=0.02), f"Aspect ratio mismatch: {aspect_ratio} != {width / height}, " \
    #                                        f"img_width: {img_width}, img_height: {img_height}"
    # image = cv2.resize(image, dsize=(width, height), interpolation= cv2.INTER_LINEAR)
    print(image.shape)
    y2, y1, x2, x1 = 0, 0, 0, 0
    if y2 - y1 > x2 - x1:
        # box_size = (y2 - y1) / image.shape[0]
        # box_size = 1 / box_size * (height / image.shape[0])
        box_size = height / (y2 - y1)
    else:
        # box_size = (x2 - x1) / image.shape[1]
        # box_size = 1 / box_size * (width / image.shape[1])
        box_size = width / (x2 - x1)

    box_size *= 1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    x_offset = (width - 1.0) / 2.0 - center_x * box_size
    y_offset = (height - 1.0) / 2.0 - center_y * box_size
    warp_matrix = np.float32([[box_size, 0, x_offset], [0, box_size, y_offset]])
    source_image_warped = cv2.warpAffine(image, warp_matrix, (width, height)).astype(
        np.float32
    )

    return source_image_warped


def invert_warp_to_paste(image, crop, crop_mask, warp_mat, H, W):
    warp_mat_T = np.linalg.inv(warp_mat)
    crop_unwarped = cv2.warpAffine(crop, warp_mat_T[:2, ...], (W, H))
    crop_mask_unwarped = cv2.warpAffine(crop_mask.astype(np.float32), warp_mat_T[:2, ...], (W, H)).astype(np.uint8)
    if len(crop_unwarped.shape) < 3:
        crop_unwarped = crop_unwarped[..., None]
    part1 = crop_unwarped * (crop_mask_unwarped[..., None])
    part2 = image * (1 - crop_mask_unwarped[..., None])
    pasted_image = part1 + part2
    return pasted_image.astype(np.uint8)

#            mask_no_headcrop = invert_warp_to_paste(mask[..., None]*255, np.zeros_like(head_crop)*255,
#                                                                  np.zeros_like(warped_head_mask),
#                                                                  warp_mat3d_to_head, H, W)


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


def get_best_human_ind(frame_alpha_data):
    biggest_box_size = 0
    best_human_ind = 0
    for human_ind, human in enumerate(frame_alpha_data['humans']):
        x1, y1, x2, y2 = human['bbox']
        bbox_size = (y2 - y1) * (x2 - x1) / 100
        if bbox_size > biggest_box_size:
            biggest_box_size = bbox_size
            best_human_ind = human_ind
    return best_human_ind


def enlarge_bbox(bbox, img_h, img_w, x_diff, y_diff):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1

    x1 = max(x1 - w * x_diff, 0)
    y1 = max(y1 - h * y_diff, 0)
    x2 = min(x2 + w * x_diff, img_w)
    y2 = min(y2 + h * y_diff, img_h)
    return np.array([x1, y1, x2, y2]).astype(int)


def get_warp_mat_to_crop(source_image, body_kp, face_kp, face_pca_head_bbox, enlarge_by=0.7, target_scale=210, target_center_y=255):
    if body_kp[0, 2] > 0.1 and body_kp[1, 2] > 0.1 and body_kp[2, 2] > 0.1:
        print(f"warp mat to face by tddfa")
        img_face, warp_mat3d = get_3ddfa_face_crop(face_kp, source_image, target_scale, target_center_y)
        return img_face, warp_mat3d
    else:
        print(f"warp mat to face by alphapose")
        target_width = 512
        target_height = 512
        x1, y1, x2, y2 = enlarge_bbox(face_pca_head_bbox, source_image.shape[0], source_image.shape[1], enlarge_by,
                                      enlarge_by)

        if y2 - y1 > x2 - x1:
            scale = target_height / (y2 - y1)
        else:
            scale = target_width / (x2 - x1)

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        x_offset = (target_width - 1.0) / 2.0 - center_x * scale
        y_offset = (target_height - 1.0) / 2.0 - center_y * scale

        warp_matrix = np.float32([[scale, 0, x_offset], [0, scale, y_offset]])
        img_face = cv2.warpAffine(source_image, warp_matrix, (target_width, target_height))
        warp_mat3d = np.vstack([warp_matrix, [0, 0, 1]])
        return img_face, warp_mat3d


def get_warp_im(img, warp_mat3d, dst_width=512, dst_height=512):
    img_trans = cv2.warpAffine(img, warp_mat3d[:2], (dst_width, dst_height))
    return img_trans


def get_crop_by_head(source_image, body_kp, face_kp, face_pca_head_bbox, enlarge_by=0.7, target_scale=210, target_center_y=255):
    if body_kp[0, 2] > 0.1 and body_kp[1, 2] > 0.1 and body_kp[2, 2] > 0.1:
        print(f"cropping face by tddfa")
        # face_kp = np.array(human_face['landmark']).T
        # new face_kp (shifted by alphapose bbox)
        img_face, warp_mat3d = get_3ddfa_face_crop(face_kp, source_image, target_scale, target_center_y)
        return img_face, warp_mat3d
    else:
        target_width = 512
        target_height = 512
        print(f"cropping face by alphapose")
        # face_pca_head_bbox = np.array(human_face['head_bounding_box'])
        x1, y1, x2, y2 = enlarge_bbox(face_pca_head_bbox, source_image.shape[0], source_image.shape[1], enlarge_by,
                                      enlarge_by)

        if y2 - y1 > x2 - x1:
            scale = target_height / (y2 - y1)
        else:
            scale = target_width / (x2 - x1)

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        x_offset = (target_width - 1.0) / 2.0 - center_x * scale
        y_offset = (target_height - 1.0) / 2.0 - center_y * scale

        warp_matrix = np.float32([[scale, 0, x_offset], [0, scale, y_offset]])
        img_face = cv2.warpAffine(source_image, warp_matrix, (target_width, target_height))
        warp_mat3d = np.vstack([warp_matrix, [0, 0, 1]])
        # img_face = source_image[y1:y2, x1:x2]
        return img_face, warp_mat3d


def get_alpha_and_face(dir_path):
    src_info_alpha_path = os.path.join(dir_path, 'info_alpha_key.json')
    with open(src_info_alpha_path, 'r') as f:
        src_video_alpha_info = json.load(f)

    src_info_face_pca_path = os.path.join(dir_path, 'face_3ddfav2.json')
    with open(src_info_face_pca_path, 'r') as f:
        src_info_face_pca = json.load(f)
    return src_video_alpha_info, src_info_face_pca


def main():
    seed = 4642326
    ddim_steps = 100

    seed_everything(seed)
    # face_crop_size = 256
    # face_crop_scale = 2.0
    # H, W, C, f = 960, 640, 4, 8
    # H, W, C, f = 1280 + 64 + 64, 960, 4, 8
    H, W, C, f = 960, 640, 4, 8
    face_crop_size = 512
    face_crop_scale = 1.5
    use_head = True
    use_face = True
    use_init_head = False
    body_denoising_strength = 0.5
    head_denoising_strength = 0.65
    face_denoising_strength = 0.5  # 0.35
    scale = 20
    # prompt = "An apartment complex in the city"
    # prompt = "a baby sitting on a bench"
    # prompt = "a green bench in a park"
    # prompt = "Emma Watson, studio lightning, realistic, fashion photoshoot, asos, perfect face, symmetric face"
    prompt = "qwv man, german, european, caucasian, bright skin"
    negative_prompt = "makeup, artistic, photoshop, painting, artstation, art, ugly, unrealistic, imaginative, african, blurry"

    config = OmegaConf.load("../configs/stable-diffusion/v1-inference.yaml")
    model_name = "avi_dual_face_3ddfa_qwv"
    model = load_model_from_config(config, f"../../dreambooth_sd/stable_diffusion_weights/{model_name}/model.ckpt")
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
    num_ims_dir1 = len(image_list1)

    src_dir2 = '../assets/sample_images/fashion_images/target_ims/'
    src_video_alpha_info2, src_info_face_pca2 = get_alpha_and_face(src_dir2)
    image_list2 = ["terminal_cauc_back",  "terminal_cauc_full", "terminal_cauc_side",
                  "terminal_dark", "terminal_dark_back", "terminal_dark_big"]
    use_face2 = [0, 1, 1, 1, 0, 1]
    # image_list2 = ["terminal_dark", "terminal_dark_back", "terminal_dark_big"]
    # use_face2 = [1, 0, 1]
    # image_list2 = ["terminal_dark_back"]
    # use_face2 = [0]

    image_list = image_list1 + image_list2
    src_video_alpha_info, src_info_face_pca = src_video_alpha_info1, src_info_face_pca1
    src_video_alpha_info.update(src_video_alpha_info2)
    src_info_face_pca.update(src_info_face_pca2)
    use_faces = use_face1 + use_face2

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

        body_kp = warped_body_kp
        face_kp = warped_face_kp
        face_pca_head_bbox = warped_face_pca_head_bbox

        target_scale_face = target_scales[0]
        enlarge_by_face = enlarge_bys[0]
        target_scale_head = target_scales[1]
        enlarge_by_head = enlarge_bys[1]

        if use_head or use_init_head:
            head_image, warp_mat3d_to_head = get_crop_by_head(source_image, body_kp, face_kp,
                                                              face_pca_head_bbox, enlarge_by=enlarge_by_head,
                                                              target_scale=target_scale_head, target_center_y=255)
            head_crop_mask = get_warp_im(warped_head_mask / 255, warp_mat3d_to_head)
            head_crop_mask = dilate_mask(head_crop_mask, kernel_size=mask_dilation * 8)

            head_crop_clothes_mask = get_warp_im(clothes_mask / 255, warp_mat3d_to_head)
            head_crop_mask = (1 - head_crop_clothes_mask) * head_crop_mask  # restricted

            # head_crop_skin_mask = get_warp_im(mask, warp_mat3d_to_head)
            # head_crop_skin_mask = np.clip(head_crop_skin_mask+head_crop_mask, 0, 1)
            # head_crop_skin_mask = (1 - head_crop_clothes_mask) * head_crop_skin_mask  # restricted

            # pasted_head_body = invert_warp_to_paste(source_image_warped, head_image,
            #                                             head_crop_mask,
            #                                             warp_mat3d_to_head, H, W)
            if use_face and use_face_cur:
                face_image, warp_mat3d_to_face = get_crop_by_head(source_image, body_kp, face_kp,
                                                                   face_pca_head_bbox, enlarge_by=enlarge_by_face,
                                                                   target_scale=target_scale_face, target_center_y=255)

                face_crop_mask = get_warp_im(warped_head_mask / 255, warp_mat3d_to_face)
                face_crop_mask = dilate_mask(face_crop_mask, kernel_size=mask_dilation * 8)
                face_crop_clothes_mask = get_warp_im(clothes_mask / 255, warp_mat3d_to_face)
                face_crop_mask = (1 - face_crop_clothes_mask) * face_crop_mask  # restricted
                # pasted_face_body = invert_warp_to_paste(source_image_warped, face_image,
                #                                             face_crop_mask,
                #                                             warp_mat3d_to_face, H, W)

                # face_crop_of_pasted_head_body = get_warp_im(pasted_head_body, warp_mat3d_to_face)  # before and after inpaint of face..
                # face_warp_of_pasted_head_body = get_warp_im(head_image, np.linalg.inv(warp_mat3d_to_head)@warp_mat3d_to_face)
                # pasted_face_headbody = invert_warp_to_paste(pasted_head_body, face_crop_of_pasted_head_body,
                #                                             face_crop_mask,
                #                                             warp_mat3d_to_face, H, W)

        # for scale in tqdm([20, 30], 'scales'):
        # for face_denoising_strength in tqdm([0.5, 0.65, 0.8], 'noises'):
        # for body_denoising_strength in tqdm([0.6, 0.7, 0.8, 0.9], 'body noises'):  # [0.3, 0.35, 0.4, 0.45, 0.5]

        if use_init_head:
            _, recon_crop_head = inpaint_image(head_image, head_crop_mask, model, prompt, negative_prompt,
                                                ddim_steps=ddim_steps, denoising_strength=head_denoising_strength,
                                                cfg_scale=scale, device=device)
            source_image_pasted_head = invert_warp_to_paste(source_image.copy(), recon_crop_head[0],
                                                                 head_crop_mask,
                                                                 warp_mat3d_to_head, H, W)
            _, recon_image = inpaint_image(source_image_pasted_head.copy(), mask.copy(), model, prompt, negative_prompt,
                                              ddim_steps=ddim_steps, denoising_strength=body_denoising_strength,
                                              cfg_scale=scale, device=device)
        else:
            _, recon_image = inpaint_image(source_image.copy(), mask.copy(), model, prompt, negative_prompt,
                                           ddim_steps=ddim_steps, denoising_strength=body_denoising_strength,
                                           cfg_scale=scale, device=device)
        # _, recon_image_sqr = inpaint_image(recon_image[0].copy(), mask.copy(), model, prompt, negative_prompt,
        #                                   ddim_steps=ddim_steps, denoising_strength=0.4,
        #                                   cfg_scale=scale, device=device)
        # _, recon_image_sqr3 = inpaint_image(recon_image_sqr[0].copy(), mask.copy(), model, prompt, negative_prompt,
        #                                   ddim_steps=ddim_steps, denoising_strength=0.4,
        #                                   cfg_scale=scale, device=device)
        if use_head:
            # _, recon_face_crop_head = inpaint_image(head_image, head_crop_mask, model, prompt, negative_prompt,
            #                                     ddim_steps=ddim_steps, denoising_strength=denoising_strength,
            #                                     cfg_scale=scale, device=device)

            head_crop = get_warp_im(recon_image[0].copy(), warp_mat3d_to_head)
            _, recon_face_crop_bodyhead = inpaint_image(head_crop, head_crop_mask, model, prompt, negative_prompt,
                                                ddim_steps=ddim_steps, denoising_strength=head_denoising_strength,
                                                cfg_scale=scale, device=device)
            recon_image_pasted_headtobody = invert_warp_to_paste(recon_image[0], recon_face_crop_bodyhead[0],
                                                                 head_crop_mask,
                                                                 warp_mat3d_to_head, H, W)

            plt.imshow(head_crop)
            plt.show()
            plt.imshow(recon_face_crop_bodyhead[0])
            plt.show()

            if use_face and use_face_cur:
                # _, recon_face_crop_face = inpaint_image(face_image, face_crop_mask, model, prompt, negative_prompt,
                #                                     ddim_steps=ddim_steps, denoising_strength=denoising_strength,
                #                                     cfg_scale=scale, device=device)
                # face_crop = get_warp_im(recon_image[0].copy(), warp_mat3d_to_face)
                # _, recon_face_crop_bodyface = inpaint_image(face_crop, face_crop_mask, model, prompt, negative_prompt,
                #                                     ddim_steps=ddim_steps, denoising_strength=denoising_strength,
                #                                     cfg_scale=scale, device=device)

                face_crop_from_headbody = get_warp_im(recon_image_pasted_headtobody, warp_mat3d_to_face)
                _, recon_face_crop_bodyheadface = inpaint_image(face_crop_from_headbody, face_crop_mask, model, prompt,
                                                                negative_prompt,
                                                                ddim_steps=ddim_steps,
                                                                denoising_strength=face_denoising_strength,
                                                                cfg_scale=scale, device=device)
                recon_image_pasted_facetoheadbody = invert_warp_to_paste(recon_image_pasted_headtobody,
                                                                         recon_face_crop_bodyheadface[0],
                                                                         face_crop_mask, warp_mat3d_to_face, H, W)

        use_upscale = True
        if use_upscale:
            use_upscale_face = False
            if use_upscale_face:
                head_mask_source = invert_warp_to_paste(np.zeros_like(mask[..., None]),
                                                        head_crop_mask * 255,
                                                        head_crop_mask,
                                                        warp_mat3d_to_head, H, W)[..., 0]

                mask_body_no_facecrop = invert_warp_to_paste(head_mask_source[..., None] * 255, np.zeros_like(face_crop_from_headbody) * 255,
                                                        np.ones_like(face_crop_mask),
                                                        warp_mat3d_to_face, H, W)[..., 0]


                mask_head_no_facecrop = get_warp_im(mask_body_no_facecrop, warp_mat3d_to_head)
                head_crop_ofpasted = get_warp_im(recon_image_pasted_facetoheadbody, warp_mat3d_to_head)

                _, upscaled_head_recon_image = inpaint_image(head_crop_ofpasted.copy(),
                                                             mask_no_headcrop.copy() / 255, model, prompt,
                                                             negative_prompt,
                                                             ddim_steps=ddim_steps, denoising_strength=0.3,
                                                             cfg_scale=scale, device=device)

            mask_no_headcrop = invert_warp_to_paste(mask[..., None]*255, np.zeros_like(head_crop)*255,
                                                                 np.ones_like(head_crop_mask),
                                                                 warp_mat3d_to_head, H, W)[..., 0]
            # mask_no_head = invert_warp_to_paste(mask[..., None]*255, np.zeros_like(head_crop)*255,
            #                                                      head_crop_mask,
            #                                                      warp_mat3d_to_head, H, W)
            head_crop_ofpasted = get_warp_im(recon_image_pasted_facetoheadbody, warp_mat3d_to_head)


            _, upscaled_body_recon_image05 = inpaint_image(recon_image_pasted_facetoheadbody.copy(), mask_no_headcrop.copy()/255, model, prompt, negative_prompt,
                                           ddim_steps=ddim_steps, denoising_strength=0.5,
                                           cfg_scale=scale, device=device)
            _, upscaled_body_recon_image03 = inpaint_image(recon_image_pasted_facetoheadbody.copy(), mask_no_headcrop.copy()/255, model, prompt, negative_prompt,
                                           ddim_steps=ddim_steps, denoising_strength=0.3,
                                           cfg_scale=scale, device=device)
            _, upscaled_body_recon_image01 = inpaint_image(recon_image_pasted_facetoheadbody.copy(), mask_no_headcrop.copy()/255, model, prompt, negative_prompt,
                                           ddim_steps=ddim_steps, denoising_strength=0.1,
                                           cfg_scale=scale, device=device)
            head_crop_ofpasted5 = get_warp_im(upscaled_body_recon_image05[0], warp_mat3d_to_head)
            head_crop_ofpasted3 = get_warp_im(upscaled_body_recon_image03[0], warp_mat3d_to_head)
            head_crop_ofpasted1 = get_warp_im(upscaled_body_recon_image01[0], warp_mat3d_to_head)

            plt.imshow(recon_image_pasted_facetoheadbody)
            plt.show()
            plt.imshow(upscaled_body_recon_image05[0])
            plt.show()
            plt.imshow(upscaled_body_recon_image03[0])
            plt.show()
            plt.imshow(upscaled_body_recon_image01[0])
            plt.show()
            plt.imshow(mask_no_headcrop)
            plt.show()
            plt.imshow(recon_face_crop_bodyhead[0])
            plt.show()
            plt.imshow(head_crop_ofpasted5)
            plt.show()
            plt.imshow(head_crop_ofpasted3)
            plt.show()
            plt.imshow(head_crop_ofpasted1)
            plt.show()

            blended = recon_face_crop_bodyhead[0] * head_crop_mask[..., None] + head_crop_ofpasted5 * (1 - head_crop_mask[..., None])
            plt.imshow(blended.astype(np.uint8))
            plt.show()
        # # w = 3
        # # # recon_face_crop_head[:, :w] = 0
        # # # recon_face_crop_head[:, -w:] = 0
        # # # recon_face_crop_head[:, :, :w] = 0
        # # # recon_face_crop_head[:, :, -w:] = 0
        # # # recon_face_crop_face[:, :w] = 0
        # # # recon_face_crop_face[:, -w:] = 0
        # # # recon_face_crop_face[:, :, :w] = 0
        # # # recon_face_crop_face[:, :, -w:] = 0
        # # recon_face_crop_bodyhead[:, :w] = 0
        # # recon_face_crop_bodyhead[:, -w:] = 0
        # # recon_face_crop_bodyhead[:, :, :w] = 0
        # # recon_face_crop_bodyhead[:, :, -w:] = 0
        # # # recon_face_crop_bodyface[:, :w] = 0
        # # # recon_face_crop_bodyface[:, -w:] = 0
        # # # recon_face_crop_bodyface[:, :, :w] = 0
        # # # recon_face_crop_bodyface[:, :, -w:] = 0
        # # recon_face_crop_bodyheadface[:, :w] = 0
        # # recon_face_crop_bodyheadface[:, -w:] = 0
        # # recon_face_crop_bodyheadface[:, :, :w] = 0
        # # recon_face_crop_bodyheadface[:, :, -w:] = 0
        im_ind = 0
        if use_init_head:
            source_image_pasted_head = cv2.cvtColor(source_image_pasted_head, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{im_ind}_{body_denoising_strength}_{head_denoising_strength}_"
                                                  f"{face_denoising_strength}_{scale}_{ddim_steps}_{H}_{W}_facepaste.png"), source_image_pasted_head)
            im_ind += 1
        rec_img_body = cv2.cvtColor(recon_image[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{im_ind}_{body_denoising_strength}_{head_denoising_strength}_"
                                              f"{face_denoising_strength}_{scale}_{ddim_steps}_{H}_{W}_body_noise05.png"), rec_img_body)
        im_ind += 1
        # rec_img_bodysqr = cv2.cvtColor(recon_image_sqr[0], cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{body_denoising_strength}_{face_denoising_strength}_{scale}_{ddim_steps}_{H}_{W}_bodysqr_noise0504.png"), rec_img_bodysqr)
        # rec_img_bodysqr3 = cv2.cvtColor(recon_image_sqr3[0], cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{body_denoising_strength}_{face_denoising_strength}_{scale}_{ddim_steps}_{H}_{W}_bodysqr3_noise050404.png"), rec_img_bodysqr3)

        if use_head:
            rec_img_pasted_headtobody = cv2.cvtColor(recon_image_pasted_headtobody, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(
                outputs_dir, f"{image_name}_{im_ind}_{body_denoising_strength}_{head_denoising_strength}_"
                             f"{face_denoising_strength}_{scale}_{ddim_steps}_"
                             f"{H}_{W}_pasted_bodyhead.png"), rec_img_pasted_headtobody)
            im_ind += 1
            if use_face and use_face_cur:
                rec_img_pasted_facetoheadbody = cv2.cvtColor(recon_image_pasted_facetoheadbody, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{im_ind}_{body_denoising_strength}_"
                                                      f"{head_denoising_strength}_{face_denoising_strength}"
                                                      f"_{scale}_{ddim_steps}_{H}_{W}_pasted_bodyheadface.png"),
                            rec_img_pasted_facetoheadbody)
                im_ind += 1
        # cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{body_denoising_strength}_{denoising_strength}_{scale}_
        #   {ddim_steps}_{H}_{W}_pasted_head.png"), rec_img_pasted_1)
        # cv2.imwrite(os.path.join(outputs_dir, f"{image_name}_{body_denoising_strength}_{denoising_strength}_{scale}_
        #   {ddim_steps}_{H}_{W}_pasted_face.png"), rec_img_pasted_2)


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


