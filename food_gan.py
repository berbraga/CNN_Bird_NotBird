"""
Simple DCGAN for generating food images.
- Train: python food_gan.py --mode train --data_dir data/food --output_dir outputs/food_gan
- Sample: python food_gan.py --mode sample --checkpoint outputs/food_gan/generator_last.pt --num_samples 16
"""
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="DCGAN to generate food images")
    parser.add_argument("--mode", choices=["train", "sample"], default="train", help="Run training or just sample from a checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/food", help="Directory with food images (any folder structure)")
    parser.add_argument("--output_dir", type=str, default="outputs/food_gan", help="Where to save checkpoints and samples")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--image_size", type=int, default=64, help="Square image size after resize/crop")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent vector size for generator input")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam")
    parser.add_argument("--save_every", type=int, default=400, help="Save sample grid every N iterations")
    parser.add_argument("--num_samples", type=int, default=16, help="How many images to sample when mode=sample")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to generator checkpoint for sampling/finetuning")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()


class FlatImageFolder(Dataset):
    """Loads every image under root recursively, ignoring labels."""

    def __init__(self, root: str, transform=None, extensions: List[str] = None):
        self.root = Path(root)
        self.transform = transform
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
        self.paths = []
        for ext in extensions:
            self.paths.extend(self.root.rglob(f"*{ext}"))
        self.paths = [p for p in self.paths if p.is_file()]
        if not self.paths:
            raise RuntimeError(f"No images found in {self.root.resolve()}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def make_dataloader(data_dir: str, image_size: int, batch_size: int) -> DataLoader:
    transform = T.Compose([
        T.Resize(image_size + 4),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = FlatImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, channels: int = 3, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, channels: int = 3, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def save_samples(generator: Generator, epoch: int, step: int, output_dir: Path, device: torch.device, latent_dim: int, num_samples: int = 16):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
        fake = generator(noise)
        grid = vutils.make_grid(fake, padding=2, normalize=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"samples_e{epoch:03d}_s{step:06d}.png"
    vutils.save_image(grid, out_path)
    generator.train()
    print(f"Saved samples to {out_path}")


def train(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = make_dataloader(args.data_dir, args.image_size, args.batch_size)

    netG = Generator(args.latent_dim).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    global_step = 0
    for epoch in range(args.epochs):
        for real in dataloader:
            real = real.to(device)
            b_size = real.size(0)

            label_real = torch.ones(b_size, device=device)
            label_fake = torch.zeros(b_size, device=device)

            # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            output_real = netD(real).view(-1)
            lossD_real = criterion(output_real, label_real)
            lossD_real.backward()

            noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach()).view(-1)
            lossD_fake = criterion(output_fake, label_fake)
            lossD_fake.backward()
            optimizerD.step()

            # Update Generator: maximize log(D(G(z)))
            netG.zero_grad()
            output_fake_for_G = netD(fake).view(-1)
            lossG = criterion(output_fake_for_G, label_real)
            lossG.backward()
            optimizerG.step()

            if global_step % args.save_every == 0:
                save_samples(netG, epoch, global_step, output_dir, device, args.latent_dim)

            if global_step % 2 == 0:
                d_loss = (lossD_real + lossD_fake).item()
                print(f"Epoch [{epoch+1}/{args.epochs}] Step {global_step} | D loss: {d_loss:.4f} | G loss: {lossG.item():.4f}")

            global_step += 1

        # Save checkpoints per epoch
        torch.save(netG.state_dict(), output_dir / "generator_last.pt")
        torch.save(netD.state_dict(), output_dir / "discriminator_last.pt")
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            grid = vutils.make_grid(fake, padding=2, normalize=True)
            vutils.save_image(grid, output_dir / f"fixed_e{epoch+1:03d}.png")
        print(f"Epoch {epoch+1} completed. Checkpoint saved.")


def sample(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required when mode=sample")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    netG = Generator(args.latent_dim).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(state)
    netG.eval()

    with torch.no_grad():
        noise = torch.randn(args.num_samples, args.latent_dim, 1, 1, device=device)
        fake = netG(noise)
        grid = vutils.make_grid(fake, padding=2, normalize=True)
    out_path = output_dir / "samples.png"
    vutils.save_image(grid, out_path)
    print(f"Saved samples to {out_path}")


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
