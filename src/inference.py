from typing import Optional
import torch
from config import Config
from pathlib import Path
import sys
import os
top_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(top_dir))

from Unet import UNet
from config import Config
from plot import *
import torch
from ddpm_scheduler import DDPMScheduler
from tqdm import tqdm

configpath = os.path.dirname(__file__)+"/config.yaml"
savepath = os.path.dirname(__file__)+"/../save/"

def inference(model:UNet, scheduler:DDPMScheduler, images: int, config: 'Config', noise: Optional[torch.Tensor] = None):
    # 选择使用固定的噪声来推理，还是随机的噪声来推理
    if noise is None:
        noisy_sample = torch.randn((images, config.input_channels, config.train_image_size,  config.train_image_size)).to(config.device)
    else:
        noisy_sample = noise
    progress_bar = tqdm(total=len(scheduler.timesteps))
    for t in scheduler.timesteps:
        with torch.no_grad():   # 不加入这一行显存会溢出
            noisy_pred = model(noisy_sample, t[None].to(config.device))
            noisy_sample = scheduler.step(noisy_pred, t, noisy_sample)  # type: ignore
        progress_bar.update(1)
        logs = {"t": t.item()}
        progress_bar.set_postfix(**logs)
    return noisy_sample

if __name__ == "__main__":
    config = Config(configpath)
    model = UNet(config).to(config.device)
    model.eval()

    scheduler = DDPMScheduler(config)

    # 读取模型
    checkpoint = torch.load(savepath + "ep10.pt")
    save = checkpoint['model_state_dict']
    print(save[0:5])
    model.load_state_dict(checkpoint['model_state_dict']) # need strict=False
    image = inference(model, scheduler, config.num_inference_images, config)
    image = (image / 2 + 0.5).clamp(0, 1)
    plot_images(image, save_dir="ans") 