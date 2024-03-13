from typing import TYPE_CHECKING
import numpy
import torch

if TYPE_CHECKING:
    from config import Config

class Scheduler:
    def __init__(self, config: 'Config') -> None:
        self.config = config

        self.num_train_timesteps: int = self.config.num_train_timesteps
        self.num_inference_steps: int = self.config.num_inference_timesteps
        self.set_timesteps()

        # 创建beta数组
        self.beta_start: float = self.config.beta_start
        self.beta_end: float = self.config.beta_end
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32).to(self.config.device) # step 个数

        self.alphas = 1.0 - self.betas      # alpha = 1 - beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)     # α_bar

    def set_timesteps(self):
        step = self.num_train_timesteps // self.num_inference_steps
        timesteps = numpy.arange(self.num_train_timesteps, 0, -step)
        self.timesteps = torch.from_numpy(timesteps).to(self.config.device)

    def add_noise(self, image: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # x_t = √(α_t)x_0 + √(1-α_t) ε
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])     # √α_bar_t
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(image.shape): #一定要加，否则会维度不匹配
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = torch.sqrt((1 - self.alphas_cumprod[timesteps]))     # √1-α_bar_t
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(image.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * image + sqrt_one_minus_alpha_prod * noise

        return noisy_samples
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.config.device).long() # 生成batch_size个[0, num_train_timesteps)的随机长整型
    
    def prev_timestep(self, timestep: torch.Tensor):
        return timestep - self.num_train_timesteps // self.num_inference_steps
    
