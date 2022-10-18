import k_diffusion as K
import torch
from tqdm import trange


class CFGCompVisDenoiser(K.external.CompVisDenoiser):
    def __init__(self, model, quantize=False, device='cpu', batch_cond_uncond=True):
        super(CFGCompVisDenoiser, self).__init__(model, quantize=False, device='cpu')
        self.batch_cond_uncond = batch_cond_uncond


    def get_eps(self, x, t, cond=None, uncond=None, cfg_scale=1.0, **kwargs):
        if cfg_scale == 1.0:
            return self.inner_model.apply_model(x, t, cond=cond)
        if uncond is None:
            uncond = self.inner_model.get_learned_conditioning([""] * x.shape[0])
        if cfg_scale == 0.0:
            return self.inner_model.apply_model(x, t, cond=uncond)
        assert cond is not None
        if self.batch_cond_uncond:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            cond_in = torch.cat([uncond, cond])
            eps = self.inner_model.apply_model(x_in, t_in, cond=cond_in)
            eps_uncond, eps_cond = eps.chunk(2)
        else:
            eps_cond = self.inner_model.apply_model(x, t, cond=cond)
            eps_uncond = self.inner_model.apply_model(x, t, cond=uncond)
        return eps_uncond + (eps_cond - eps_uncond) * cfg_scale


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
    return x


def get_progress_bar(n_iter, desc=""):
    it = iter(trange(n_iter, desc=desc))
    return lambda x: next(it)


