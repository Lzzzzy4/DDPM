from Unet import UNet
from imageload import ImageDataset
import torch
import torch.nn as nn
from config import Config
from tqdm import tqdm
from imageload import ImageDataset
import os
from scheduler import Scheduler
from plot import plot_images
from torch.utils.data import DataLoader
from DFUnet import DFUNet
from mnist_data import MNISTData

path = os.path.dirname(__file__)+"/../data/1.jpg"
configpath = os.path.dirname(__file__)+"/config.yaml"
savepath = os.path.dirname(__file__)+"/../save/"

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("cuda is available")
    print("device_count",torch.cuda.device_count())
    config = Config(configpath)
    # model = UNet(config).to(config.device)
    model = DFUNet(config).to(config.device)
    scheduler = Scheduler(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    training_data = MNISTData(config, r"dataset", return_label=True)
    train_dataloader = DataLoader(training_data, batch_size=config.batch, shuffle=True)
    
    # 显示
    # train_image = ImageDataset(path, config)[0]
    # train_image = train_image[None, ...].to(config.device)
    # timesteps = scheduler.sample_timesteps(batch_size=1).to(config.device)
    # noise = torch.randn(train_image.shape).to(config.device)
    # noisy_image = scheduler.add_noise(image=train_image, noise=noise, timesteps=timesteps)

    # plot_images((train_image / 2 + 0.5).clamp(0, 1), fig_titles="original image", save_dir=config.proj_name)
    # plot_images((noisy_image / 2 + 0.5).clamp(0, 1), fig_titles="noisy image", save_dir=config.proj_name)

    # progress_bar = tqdm(total=config.epochs)
    for ep in range(config.epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        model.train()
        for image, _ in train_dataloader:
            batch = image.shape[0]
            timesteps = scheduler.sample_timesteps(batch)
            # image = train_image
            noise = torch.randn(image.shape).to(config.device)
            noisy_image = scheduler.add_noise(image=image, noise=noise, timesteps=timesteps)

            # pred = model(noisy_image, timesteps)[0]
            pred = model(noisy_image, timesteps)
            # print(pred.shape, image.shape, noise.shape, noisy_image.shape)
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
            }, f = savepath + "ep" + str(ep+1) + ".pt")

            # # 采样一些图片
            # if (ep+1) % config.sample_period == 0:
            #     model.eval()
            #     labels = torch.randint(0, 9, (config.num_inference_images, 1)).to(config.device)
            #     image = inference(model, scheduler, config.num_inference_images, config, label=labels)
            #     image = (image / 2 + 0.5).clamp(0, 1)
            #     plot_images(image, save_dir=config.proj_name, titles=labels.detach().tolist())
            #     model.train()
        


