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
        sqrt = torch.sqrt(self.alphas_cumprod[timesteps]).flatten() # 扁平化
        # while len(sqrt_alpha_prod.shape) < len(image.shape):
        #     sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt1 = torch.sqrt(1 - self.alphas_cumprod[timesteps]).flatten()
        # sqrt.cuda()
        # sqrt1.cuda()
        return sqrt * image + sqrt1 * noise
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.config.device).long() # 生成batch_size个[0, num_train_timesteps)的随机长整型
    
    def prev_timestep(self, timestep: torch.Tensor):
        return timestep - self.num_train_timesteps // self.num_inference_steps
    
