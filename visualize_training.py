"""
Script auxiliar para visualizar o histórico de treinamento da GAN.
Uso: python visualize_training.py --history outputs/food_gan/training_history.json
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_history(history_path: Path):
    """Carrega histórico de treinamento do arquivo JSON."""
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_full_training(history: dict, output_dir: Path):
    """Plota visualização completa do treinamento."""
    steps = history['steps']
    d_losses = history['d_losses']
    g_losses = history['g_losses']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise Completa do Treinamento da GAN', fontsize=18, fontweight='bold')
    
    # Gráfico 1: Losses completas
    axes[0, 0].plot(steps, d_losses, 'b-', linewidth=1, alpha=0.6, label='Discriminador')
    axes[0, 0].plot(steps, g_losses, 'r-', linewidth=1, alpha=0.6, label='Gerador')
    axes[0, 0].set_xlabel('Iteração', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Evolução das Losses (Todas as Iterações)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=11)
    
    # Gráfico 2: Média móvel
    window = min(100, len(d_losses) // 10)
    if len(d_losses) >= window:
        d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        
        axes[0, 1].plot(smooth_steps, d_smooth, 'b-', linewidth=2, alpha=0.8, label='Discriminador')
        axes[0, 1].plot(smooth_steps, g_smooth, 'r-', linewidth=2, alpha=0.8, label='Gerador')
        axes[0, 1].set_xlabel('Iteração', fontsize=12)
        axes[0, 1].set_ylabel('Loss (média móvel)', fontsize=12)
        axes[0, 1].set_title(f'Tendência (Média Móvel - Janela: {window})', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=11)
    
    # Gráfico 3: Resumo por época
    if 'epochs' in history and history['epochs']:
        epochs = history['epochs']
        epoch_d_losses = history['epoch_d_losses']
        epoch_g_losses = history['epoch_g_losses']
        
        axes[1, 0].plot(epochs, epoch_d_losses, 'b-o', linewidth=2, markersize=8, label='Discriminador')
        axes[1, 0].plot(epochs, epoch_g_losses, 'r-o', linewidth=2, markersize=8, label='Gerador')
        axes[1, 0].set_xlabel('Época', fontsize=12)
        axes[1, 0].set_ylabel('Loss Média', fontsize=12)
        axes[1, 0].set_title('Loss Média por Época', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=11)
    
    # Gráfico 4: Estatísticas
    axes[1, 1].axis('off')
    stats_text = f"""
    Estatísticas do Treinamento
    
    Total de Iterações: {len(steps)}
    Total de Épocas: {len(history.get('epochs', []))}
    
    Discriminador:
      Loss Mínima: {min(d_losses):.4f}
      Loss Máxima: {max(d_losses):.4f}
      Loss Média: {np.mean(d_losses):.4f}
      Loss Final: {d_losses[-1]:.4f}
    
    Gerador:
      Loss Mínima: {min(g_losses):.4f}
      Loss Máxima: {max(g_losses):.4f}
      Loss Média: {np.mean(g_losses):.4f}
      Loss Final: {g_losses[-1]:.4f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    output_path = output_dir / "training_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Análise completa salva em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualizar histórico de treinamento da GAN")
    parser.add_argument("--history", type=str, required=True, 
                       help="Caminho para o arquivo training_history.json")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Diretório para salvar gráficos (padrão: mesmo do histórico)")
    
    args = parser.parse_args()
    
    history_path = Path(args.history)
    if not history_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {history_path}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = history_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Carregando histórico de: {history_path}")
    history = load_history(history_path)
    
    print("Gerando visualizações...")
    plot_full_training(history, output_dir)
    
    print("Visualização concluída!")


if __name__ == "__main__":
    main()

