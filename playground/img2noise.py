import torch
import numpy as np
import k_diffusion as K

from PIL import Image
from torch import autocast
from einops import rearrange, repeat
from tqdm import trange


def pil_img_to_torch(pil_img, half=False):
    image = np.array(pil_img).astype(np.float32) / 255.0
    image = rearrange(torch.from_numpy(image), 'h w c -> c h w')
    if half:
        image = image.half()
    return (2.0 * image - 1.0).unsqueeze(0)


def pil_img_to_latent(model, images, batch_size=1, device='cuda', half=True):
    if isinstance(images, Image.Image):
        images = [images]
    init_image = torch.cat([pil_img_to_torch(img, half=half).to(device) for img in images], dim=0)
    if len(init_image) == 1 and batch_size > 1:
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    if half:
        return model.get_first_stage_encoding(model.encode_first_stage(init_image.half()))
    return model.get_first_stage_encoding(model.encode_first_stage(init_image))


def find_noise_for_image(model, images, prompt, latent=None, steps=200, cond_scale=0.0, verbose=False):
    if latent is None:
        x = pil_img_to_latent(model, images, batch_size=1, device='cuda', half=True)
    else:
        x = latent
    batch_size = x.shape[0]
    with torch.no_grad():
        with autocast('cuda'):
            uncond = model.get_learned_conditioning([''] * batch_size)
            cond = model.get_learned_conditioning([prompt] * batch_size)

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast('cuda'):
            for i in trange(1, len(sigmas)):
                if cond_scale == 1.0 and False:
                    x_in = x
                    sigma_in = sigmas[i - 1] * s_in
                    cond_in = cond

                    c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
                    t = dnw.sigma_to_t(sigma_in)
                    eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                elif cond_scale == 0.0:
                    x_in = x
                    sigma_in = sigmas[i - 1] * s_in
                    cond_in = uncond

                    c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
                    t = dnw.sigma_to_t(sigma_in)
                    eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                else:
                    x_in = torch.cat([x] * 2)
                    sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                    cond_in = torch.cat([uncond, cond])

                    c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

                    t = dnw.sigma_to_t(sigma_in)

                    eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                    eps_uncond, eps_cond = eps.chunk(2)
                    eps = eps_uncond + (eps_cond - eps_uncond) * cond_scale
                # else:
                #     x_in = x
                #     sigma_in = sigmas[i - 1] * s_in
                #     cond_in = cond
                #
                #     c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
                #     t = dnw.sigma_to_t(sigma_in)
                #     eps = model.apply_model(x_in * c_in, t, cond=cond_in)

                dt = sigmas[i] - sigmas[i - 1]
                x = x + eps * dt

            c_out, c_in = dnw.get_scalings(sigmas[-1])
            return x * c_in


def find_noise_for_image_(model, img, prompt, steps=200, cond_scale=0.0, verbose=False, normalize=True):
    x = pil_img_to_latent(model, img, batch_size=1, device='cuda', half=False)

    with torch.no_grad():
        with autocast('cuda'):
            uncond = model.get_learned_conditioning([''])
            cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast('cuda'):
            for i in trange(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                # sigma_in = torch.cat([sigmas[i] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

                # if i == 1:
                #     t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                # else:
                #     t = dnw.sigma_to_t(sigma_in)
                t = dnw.sigma_to_t(sigma_in)

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

                # if i == 1:
                #     d = (x - denoised) / (2 * sigmas[i])
                # else:
                #     d = (x - denoised) / sigmas[i - 1]

                # if i == 1:
                #     d = (x - denoised) / sigmas[i]
                # else:
                #     d = (x - denoised) / sigmas[i - 1]

                d = (x - denoised) / sigmas[i]

                dt = sigmas[i] - sigmas[i - 1]
                x = x + d * dt

            if normalize:
                return (x / x.std()) * sigmas[-1]
            else:
                return x


def find_noise_for_latent(model, latent, prompt, steps=200, cond_scale=0.0, verbose=False, normalize=True):
    x = latent
    batch_size = x.shape[0]
    with torch.no_grad():
        with autocast('cuda'):
            uncond = model.get_learned_conditioning([''] * batch_size)
            cond = model.get_learned_conditioning([prompt] * batch_size)

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)
    sigmas[0] = 0.000

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast('cuda'):
            for i in trange(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                # sigma_in = torch.cat([sigmas[i] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

                # if i == 1:
                #     t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                # else:
                #     t = dnw.sigma_to_t(sigma_in)
                t = dnw.sigma_to_t(sigma_in)

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                eps_uncond, eps_cond = eps.chunk(2)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale
                eps = eps_uncond + (eps_cond - eps_uncond) * cond_scale

                # if i == 1:
                #     d = (x - denoised) / (2 * sigmas[i])
                # else:
                #     d = (x - denoised) / sigmas[i - 1]

                # if i == 1:
                #     d = (x - denoised) / sigmas[i]
                # else:
                #     d = (x - denoised) / sigmas[i - 1]

                # d = (x - denoised) / sigmas[i - 1]
                # d = (x - denoised) / sigma_in

                # d = eps

                dt = sigmas[i] - sigmas[i - 1]
                x = x + eps * dt
                # x = (x * (1 - dt ** 2).sqrt()) + d * dt

                # print(f"{i} std: {x.std()}")

            if normalize:
                return (x / x.std()) * sigmas[-1]
            else:
                c_out, c_in = dnw.get_scalings(sigmas[-1])
                return (x * c_in)



def generate_image_from_noise(model, noise, prompt, steps=200, cond_scale=0.0, verbose=False):
    x = noise

    with torch.no_grad():
        with autocast('cuda'):
            uncond = model.get_learned_conditioning([''])
            cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps)

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast('cuda'):
            for i in trange(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

                if i == 1:
                    t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                else:
                    t = dnw.sigma_to_t(sigma_in)

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                denoised_uncond, denoised_cond = (x_in - eps * c_out).chunk(2)

                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

                if i == -1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i - 1]

                dt = sigmas[i] - sigmas[i - 1]
                x = x - d * dt

            return x
