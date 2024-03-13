from scheduler import Scheduler
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import Config

class DDPMScheduler(Scheduler):
    def __init__(self, config: 'Config') -> None:
        super().__init__(config)

    def step(self, noise_pred: torch.Tensor, t: torch.Tensor, noisy_image: torch.Tensor) -> torch.Tensor:
        prev_t = self.prev_timestep(t)
        a_t = self.alphas_cumprod[t]
        a_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        b_t = 1 - a_t
        b_prev_t = 1 - a_prev_t

        alpha_t = a_t / a_prev_t
        beta_t = 1 - alpha_t

        # x_t = √α_bar_t x_0 + √1-α_bar_t ε
        # x_0 = ( x_t - √1-α_bar_t ε ) / √α_bar_t
        denoised_image = (noisy_image - torch.sqrt(b_t) * noise_pred) / torch.sqrt(a_t)
        denoised_image = denoised_image.clamp(-self.config.clip, self.config.clip) # [-1,1]

        pred_original_sample_coeff = (torch.sqrt(a_prev_t) * beta_t) / b_t
        current_sample_coeff = torch.sqrt(alpha_t) * b_prev_t / b_t
        pred_prev_image = pred_original_sample_coeff * denoised_image + current_sample_coeff * noisy_image

        variance = 0
        if t > 0 :
            z = torch.randn(noise_pred.shape).to(self.config.device)
            variance = (1 - a_prev_t) / (1 - a_t) * beta_t
            variance = torch.clamp(variance, min=1e-20)
            variance = torch.sqrt(variance) * z

        return pred_prev_image + variance