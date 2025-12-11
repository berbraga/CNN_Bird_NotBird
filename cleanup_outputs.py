"""
Script para limpar arquivos desnecessários da pasta de output.
Mantém apenas os arquivos essenciais e algumas visualizações importantes.

Uso:
  python cleanup_outputs.py --output_dir outputs/food_gan
  python cleanup_outputs.py --output_dir outputs/food_gan --keep_fixed 5  # Manter 5 fixed images
"""
import argparse
from pathlib import Path
import glob


def cleanup_outputs(output_dir: Path, keep_fixed: int = 10, keep_progress: int = 5, 
                   delete_samples: bool = True, delete_training_samples: bool = True):
    """
    Limpa arquivos desnecessários da pasta de output.
    
    Args:
        output_dir: Diretório de output
        keep_fixed: Quantas imagens fixed_eXXX.png manter (as mais recentes)
        keep_progress: Quantos gráficos training_progress manter (os mais recentes)
        delete_samples: Se True, deleta samples.png e samples_enhanced.png
        delete_training_samples: Se True, deleta todas samples_eXXX_sXXXXXX.png
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Diretório não encontrado: {output_dir}")
        return
    
    deleted_count = 0
    kept_count = 0
    
    # 1. Limpar fixed_eXXX.png (manter apenas as mais recentes)
    fixed_files = sorted(output_dir.glob("fixed_e*.png"), key=lambda x: int(x.stem.split('_e')[1]))
    if len(fixed_files) > keep_fixed:
        to_delete = fixed_files[:-keep_fixed]  # Mantém as últimas N
        for f in to_delete:
            f.unlink()
            deleted_count += 1
            print(f"Deletado: {f.name}")
        kept_count += len(fixed_files) - len(to_delete)
        print(f"Mantidas {keep_fixed} imagens fixed_eXXX.png mais recentes")
    else:
        kept_count += len(fixed_files)
        print(f"Mantidas todas as {len(fixed_files)} imagens fixed_eXXX.png")
    
    # 2. Limpar training_progress_eXXX.png (manter apenas os mais recentes)
    progress_files = sorted(output_dir.glob("training_progress_e*.png"), 
                           key=lambda x: int(x.stem.split('_e')[1]))
    if len(progress_files) > keep_progress:
        to_delete = progress_files[:-keep_progress]
        for f in to_delete:
            f.unlink()
            deleted_count += 1
            print(f"Deletado: {f.name}")
        kept_count += len(progress_files) - len(to_delete)
        print(f"Mantidos {keep_progress} gráficos training_progress mais recentes")
    else:
        kept_count += len(progress_files)
        print(f"Mantidos todos os {len(progress_files)} gráficos training_progress")
    
    # 3. Deletar samples_eXXX_sXXXXXX.png (amostras durante treinamento)
    if delete_training_samples:
        sample_files = list(output_dir.glob("samples_e*_s*.png"))
        for f in sample_files:
            f.unlink()
            deleted_count += 1
        print(f"Deletadas {len(sample_files)} imagens samples_eXXX_sXXXXXX.png")
    
    # 4. Deletar samples.png e samples_enhanced.png (podem ser regenerados)
    if delete_samples:
        for pattern in ["samples.png", "samples_enhanced.png"]:
            f = output_dir / pattern
            if f.exists():
                f.unlink()
                deleted_count += 1
                print(f"Deletado: {f.name}")
    
    # Mostrar arquivos mantidos (essenciais)
    essential_files = [
        "generator_last.pt",
        "discriminator_last.pt",
        "training_history.json",
        "training_summary.png"
    ]
    
    print("\n" + "="*60)
    print("ARQUIVOS ESSENCIAIS MANTIDOS:")
    print("="*60)
    for fname in essential_files:
        f = output_dir / fname
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"✓ {fname} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {fname} (não encontrado)")
    
    print("\n" + "="*60)
    print(f"RESUMO: {deleted_count} arquivos deletados, {kept_count} mantidos")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Limpar arquivos desnecessários do output")
    parser.add_argument("--output_dir", type=str, default="outputs/food_gan",
                       help="Diretório de output a limpar")
    parser.add_argument("--keep_fixed", type=int, default=10,
                       help="Quantas imagens fixed_eXXX.png manter (padrão: 10)")
    parser.add_argument("--keep_progress", type=int, default=5,
                       help="Quantos gráficos training_progress manter (padrão: 5)")
    parser.add_argument("--keep_samples", action="store_true",
                       help="Manter samples.png e samples_enhanced.png")
    parser.add_argument("--keep_training_samples", action="store_true",
                       help="Manter samples_eXXX_sXXXXXX.png")
    
    args = parser.parse_args()
    
    print("="*60)
    print("LIMPEZA DE ARQUIVOS DE OUTPUT")
    print("="*60)
    print(f"Diretório: {args.output_dir}")
    print(f"Manter {args.keep_fixed} imagens fixed_eXXX.png")
    print(f"Manter {args.keep_progress} gráficos training_progress")
    print("="*60 + "\n")
    
    cleanup_outputs(
        output_dir=args.output_dir,
        keep_fixed=args.keep_fixed,
        keep_progress=args.keep_progress,
        delete_samples=not args.keep_samples,
        delete_training_samples=not args.keep_training_samples
    )


if __name__ == "__main__":
    main()

