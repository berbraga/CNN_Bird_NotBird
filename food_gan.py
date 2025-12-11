"""
Simple DCGAN for generating food images.
- Train: python food_gan.py --mode train --data_dir data/food --output_dir outputs/food_gan
- Sample: python food_gan.py --mode sample --checkpoint outputs/food_gan/generator_last.pt --num_samples 16
"""
import argparse
from pathlib import Path
from typing import List
import json
import shutil

import matplotlib
matplotlib.use('Agg')  # Usar backend não-interativo
import matplotlib.pyplot as plt
import numpy as np

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
    parser.add_argument("--save_checkpoint_every", type=int, default=1, help="Save checkpoint every N epochs (default: every epoch)")
    parser.add_argument("--keep_checkpoints", type=int, default=5, help="Keep last N checkpoints (0 = keep all, default: 5)")
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


def plot_training_progress(history: dict, output_dir: Path, epoch: int):
    """Plota e salva gráficos de progresso do treinamento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    steps = history['steps']
    d_losses = history['d_losses']
    g_losses = history['g_losses']
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Progresso do Treinamento - Época {epoch}', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Loss do Discriminador
    axes[0, 0].plot(steps, d_losses, 'b-', linewidth=1.5, alpha=0.7, label='Discriminador')
    axes[0, 0].set_xlabel('Iteração')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss do Discriminador')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Gráfico 2: Loss do Gerador
    axes[0, 1].plot(steps, g_losses, 'r-', linewidth=1.5, alpha=0.7, label='Gerador')
    axes[0, 1].set_xlabel('Iteração')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss do Gerador')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Gráfico 3: Comparação de Losses (sobrepostos)
    axes[1, 0].plot(steps, d_losses, 'b-', linewidth=1.5, alpha=0.7, label='Discriminador')
    axes[1, 0].plot(steps, g_losses, 'r-', linewidth=1.5, alpha=0.7, label='Gerador')
    axes[1, 0].set_xlabel('Iteração')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Comparação de Losses')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Gráfico 4: Média móvel das losses (últimas 100 iterações)
    window = min(100, len(d_losses))
    if len(d_losses) >= window:
        d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        
        axes[1, 1].plot(smooth_steps, d_smooth, 'b-', linewidth=2, alpha=0.8, label='Discriminador (média móvel)')
        axes[1, 1].plot(smooth_steps, g_smooth, 'r-', linewidth=2, alpha=0.8, label='Gerador (média móvel)')
        axes[1, 1].set_xlabel('Iteração')
        axes[1, 1].set_ylabel('Loss (média móvel)')
        axes[1, 1].set_title(f'Média Móvel (janela de {window})')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Dados insuficientes\npara média móvel', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Média Móvel')
    
    plt.tight_layout()
    plot_path = output_dir / f"training_progress_e{epoch:03d}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de progresso salvo em: {plot_path}")


def plot_epoch_summary(history: dict, output_dir: Path):
    """Plota resumo geral do treinamento por época."""
    if 'epochs' not in history or not history['epochs']:
        return
    
    epochs = history['epochs']
    epoch_d_losses = history['epoch_d_losses']
    epoch_g_losses = history['epoch_g_losses']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Resumo do Treinamento por Época', fontsize=16, fontweight='bold')
    
    # Loss média por época - Discriminador
    axes[0].plot(epochs, epoch_d_losses, 'b-o', linewidth=2, markersize=6, label='Discriminador')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss Média')
    axes[0].set_title('Loss Média do Discriminador por Época')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Loss média por época - Gerador
    axes[1].plot(epochs, epoch_g_losses, 'r-o', linewidth=2, markersize=6, label='Gerador')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss Média')
    axes[1].set_title('Loss Média do Gerador por Época')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = output_dir / "training_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Resumo do treinamento salvo em: {plot_path}")


def save_training_history(history: dict, output_dir: Path):
    """Salva histórico de treinamento em JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.json"
    
    # Converter tensores e numpy arrays para listas
    history_serializable = {
        'steps': [int(s) for s in history['steps']],
        'd_losses': [float(l) for l in history['d_losses']],
        'g_losses': [float(l) for l in history['g_losses']],
        'epochs': [int(e) for e in history.get('epochs', [])],
        'epoch_d_losses': [float(l) for l in history.get('epoch_d_losses', [])],
        'epoch_g_losses': [float(l) for l in history.get('epoch_g_losses', [])],
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"Histórico salvo em: {history_path}")


def cleanup_old_checkpoints(output_dir: Path, current_epoch: int, keep_count: int):
    """Remove checkpoints antigos, mantendo apenas os últimos N."""
    if keep_count <= 0:
        return
    
    # Encontrar todos os checkpoints numerados
    gen_checkpoints = sorted(output_dir.glob("generator_e*.pt"), 
                            key=lambda x: int(x.stem.split('_e')[1]))
    disc_checkpoints = sorted(output_dir.glob("discriminator_e*.pt"),
                             key=lambda x: int(x.stem.split('_e')[1]))
    
    # Deletar os mais antigos
    if len(gen_checkpoints) > keep_count:
        to_delete = gen_checkpoints[:-keep_count]
        for cp in to_delete:
            try:
                cp.unlink()
            except Exception as e:
                print(f"AVISO: Não foi possível deletar {cp.name}: {e}")
    
    if len(disc_checkpoints) > keep_count:
        to_delete = disc_checkpoints[:-keep_count]
        for cp in to_delete:
            try:
                cp.unlink()
            except Exception as e:
                print(f"AVISO: Não foi possível deletar {cp.name}: {e}")


def save_checkpoint_safe(model, filepath: Path, max_retries: int = 3):
    """
    Salva checkpoint de forma segura, usando arquivo temporário primeiro.
    
    Args:
        model: Modelo PyTorch para salvar
        filepath: Caminho do arquivo final
        max_retries: Número máximo de tentativas
    """
    import time
    import shutil
    
    output_dir = filepath.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Usar arquivo temporário primeiro
    temp_path = filepath.parent / f"{filepath.name}.tmp"
    
    for attempt in range(max_retries):
        try:
            # Tentar deletar arquivo temporário antigo se existir
            if temp_path.exists():
                temp_path.unlink()
            
            # Salvar no arquivo temporário
            torch.save(model.state_dict(), temp_path)
            
            # Aguardar um pouco para garantir que o arquivo foi escrito
            time.sleep(0.1)
            
            # Verificar se o arquivo foi criado corretamente
            if not temp_path.exists():
                raise RuntimeError(f"Arquivo temporário não foi criado: {temp_path}")
            
            # Se o arquivo final existe, tentar deletá-lo primeiro
            if filepath.exists():
                try:
                    filepath.unlink()
                except PermissionError:
                    # Se não conseguir deletar, tentar renomear o antigo
                    backup_path = filepath.parent / f"{filepath.name}.backup"
                    if backup_path.exists():
                        backup_path.unlink()
                    filepath.rename(backup_path)
            
            # Renomear arquivo temporário para o final
            temp_path.rename(filepath)
            
            # Verificar se o arquivo final existe
            if not filepath.exists():
                raise RuntimeError(f"Arquivo final não foi criado: {filepath}")
            
            return True
            
        except (PermissionError, RuntimeError, OSError) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 0.5
                print(f"Tentativa {attempt + 1} falhou ao salvar {filepath.name}. "
                      f"Tentando novamente em {wait_time}s... Erro: {e}")
                time.sleep(wait_time)
            else:
                print(f"ERRO: Não foi possível salvar {filepath.name} após {max_retries} tentativas.")
                print(f"Erro: {e}")
                print("Continuando o treinamento, mas o checkpoint não foi salvo.")
                return False
    
    return False


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

    # Histórico para gráficos
    history = {
        'steps': [],
        'd_losses': [],
        'g_losses': [],
        'epochs': [],
        'epoch_d_losses': [],
        'epoch_g_losses': [],
    }

    global_step = 0
    for epoch in range(args.epochs):
        epoch_d_losses = []
        epoch_g_losses = []
        
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

            # Registrar losses
            d_loss = (lossD_real + lossD_fake).item()
            g_loss = lossG.item()
            
            epoch_d_losses.append(d_loss)
            epoch_g_losses.append(g_loss)
            
            # Salvar no histórico (a cada iteração para gráficos detalhados)
            history['steps'].append(global_step)
            history['d_losses'].append(d_loss)
            history['g_losses'].append(g_loss)

            if global_step % args.save_every == 0:
                save_samples(netG, epoch, global_step, output_dir, device, args.latent_dim)
                # Plotar progresso a cada save_every
                plot_training_progress(history, output_dir, epoch)

            if global_step % 2 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step {global_step} | D loss: {d_loss:.4f} | G loss: {g_loss:.4f}")

            global_step += 1

        # Calcular médias da época
        avg_d_loss = np.mean(epoch_d_losses)
        avg_g_loss = np.mean(epoch_g_losses)
        
        history['epochs'].append(epoch + 1)
        history['epoch_d_losses'].append(avg_d_loss)
        history['epoch_g_losses'].append(avg_g_loss)

        # Save checkpoints per epoch (com salvamento seguro)
        if (epoch + 1) % args.save_checkpoint_every == 0:
            print(f"Salvando checkpoints da época {epoch + 1}...")
            
            # Salvar checkpoint numerado
            gen_path = output_dir / f"generator_e{epoch+1:04d}.pt"
            disc_path = output_dir / f"discriminator_e{epoch+1:04d}.pt"
            
            gen_saved = save_checkpoint_safe(netG, gen_path)
            disc_saved = save_checkpoint_safe(netD, disc_path)
            
            # Sempre salvar também como "last"
            if gen_saved:
                try:
                    shutil.copy(gen_path, output_dir / "generator_last.pt")
                except Exception as e:
                    print(f"AVISO: Não foi possível copiar para generator_last.pt: {e}")
            
            if disc_saved:
                try:
                    shutil.copy(disc_path, output_dir / "discriminator_last.pt")
                except Exception as e:
                    print(f"AVISO: Não foi possível copiar para discriminator_last.pt: {e}")
            
            # Limpar checkpoints antigos se necessário
            if args.keep_checkpoints > 0:
                cleanup_old_checkpoints(output_dir, epoch + 1, args.keep_checkpoints)
            
            if gen_saved and disc_saved:
                print(f"Checkpoints salvos com sucesso.")
            else:
                print(f"AVISO: Alguns checkpoints não foram salvos. Continuando treinamento...")
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            grid = vutils.make_grid(fake, padding=2, normalize=True)
            vutils.save_image(grid, output_dir / f"fixed_e{epoch+1:03d}.png")
        
        # Plotar progresso ao final de cada época
        plot_training_progress(history, output_dir, epoch + 1)
        plot_epoch_summary(history, output_dir)
        save_training_history(history, output_dir)
        
        print(f"Epoch {epoch+1} completed. Checkpoint saved.")
        print(f"  Média D loss: {avg_d_loss:.4f} | Média G loss: {avg_g_loss:.4f}")


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

    print(f"Gerando {args.num_samples} imagens...")
    with torch.no_grad():
        noise = torch.randn(args.num_samples, args.latent_dim, 1, 1, device=device)
        fake = netG(noise)
        grid = vutils.make_grid(fake, padding=2, normalize=True)
    
    # Salvar grade de imagens
    out_path = output_dir / "samples.png"
    vutils.save_image(grid, out_path)
    print(f"Imagens salvas em: {out_path}")
    
    # Criar visualização melhorada com matplotlib
    fig, axes = plt.subplots(int(np.sqrt(args.num_samples)), int(np.sqrt(args.num_samples)), 
                            figsize=(12, 12))
    if args.num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fake_np = fake.cpu().numpy()
    for i, ax in enumerate(axes):
        if i < args.num_samples:
            img = fake_np[i].transpose(1, 2, 0)
            img = (img + 1) / 2.0  # Desnormalizar de [-1, 1] para [0, 1]
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Imagens Geradas pela GAN ({args.num_samples} amostras)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    enhanced_path = output_dir / "samples_enhanced.png"
    plt.savefig(enhanced_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualização aprimorada salva em: {enhanced_path}")


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
