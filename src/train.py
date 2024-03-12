from Unet import UNet
from imageload import ImageDataset
import torch
import torch.nn as nn
from config import Config
import tqdm.auto as tqdm
from imageload import ImageDataset
import os
from scheduler import Scheduler
from plot import plot_images

path = os.path.dirname(__file__)+"/../data/1.jpg"



if __name__ == "__main__":
    config = Config(r"config.yaml")
    model = UNet(config).to(config.device)
    # model = DFUNet(config).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_image = ImageDataset(path).image

    timesteps = Scheduler.sample_timesteps(1)
    noise = torch.randn(train_image.shape).to(config.device)
    noisy_image = Scheduler.add_noise(image=train_image, noise=noise, timesteps=timesteps)

    plot_images((train_image / 2 + 0.5).clamp(0, 1), fig_titles="original image", save_dir=config.proj_name)
    plot_images((noisy_image / 2 + 0.5).clamp(0, 1), fig_titles="noisy image", save_dir=config.proj_name)

    progress_bar = tqdm(total=len(config.epochs))
    for ep in range(config.epochs):
        model.train()
        batch = 1
        image = train_image
        noise = torch.randn(image.shape).to(config.device)
        noisy_image = Scheduler.add_noise(image=image, noise=noise, timesteps=timesteps)

        pred = model(noisy_image, timesteps)[0]
        loss = torch.nn.functional.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping, 用来防止 exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "ep": ep+1}
        progress_bar.set_postfix(**logs)

        # 保存模型
        if (ep+1) % config.save_period == 0 or (ep+1) == config.epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, r"checkpoints/model_ep" + str(ep+1))

        # # 采样一些图片
        # if (ep+1) % config.sample_period == 0:
        #     model.eval()
        #     labels = torch.randint(0, 9, (config.num_inference_images, 1)).to(config.device)
        #     image = inference(model, scheduler, config.num_inference_images, config, label=labels)
        #     image = (image / 2 + 0.5).clamp(0, 1)
        #     plot_images(image, save_dir=config.proj_name, titles=labels.detach().tolist())
        #     model.train()
        


