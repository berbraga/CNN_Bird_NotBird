# Projeto CNN Bird/NotBird

Projeto de classificação de imagens usando Redes Neurais Convolucionais (CNN) para distinguir entre pássaros e não-pássaros.

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Jupyter Notebook ou JupyterLab
- GPU NVIDIA com CUDA (opcional, mas recomendado para treinamento mais rápido)
- Arquivos ZIP com as imagens:
  - `bird.zip` - Imagens de pássaros
  - `not-bird.zip` - Imagens de não-pássaros

## 🚀 Como Executar

### 1. Instalação das Dependências

Abra um terminal e execute:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Nota:** Se você não tiver GPU NVIDIA ou quiser usar CPU, instale sem especificar o índice CUDA:

```bash
pip install torch torchvision
```

### 2. Executar o Projeto

O projeto contém vários notebooks Jupyter. Escolha um deles:

#### Opção 1: `image.ipynb` (Versão Básica)
- Carrega 2000 imagens de cada classe
- Imagens redimensionadas para 32x32
- Treinamento simples sem divisão de teste

#### Opção 2: `image_v2.ipynb` (Recomendado)
- Carrega 5000 imagens de cada classe
- Imagens redimensionadas para 32x32
- Divisão de dados: 40% treino, 60% teste
- Calcula métricas de avaliação (ACC, PRE, REV, F1)

#### Opção 3: `GAN_v1.ipynb` (Geração de Imagens)
- Implementa uma GAN (Generative Adversarial Network)
- Inclui gerador e avaliador
- Carrega imagens de 256x256

### 3. Passos para Executar

1. **Inicie o Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Abra o notebook desejado** (recomendado: `image_v2.ipynb`)

3. **Execute as células em ordem:**
   - Primeira célula: Instala dependências (se necessário)
   - Segunda célula: Carrega e processa as imagens dos arquivos ZIP
   - Terceira célula: Define a arquitetura da CNN
   - Quarta célula: Define a função de treinamento
   - Quinta célula: Treina o modelo
   - Últimas células: Avalia o modelo e calcula métricas

## 📁 Estrutura do Projeto

```
CNN_Bird_NotBird/
├── bird.zip              # Imagens de pássaros
├── not-bird.zip          # Imagens de não-pássaros
├── image.ipynb           # Versão básica
├── image_v2.ipynb        # Versão com validação (recomendado)
├── GAN_v1.ipynb          # Versão com GAN
├── data/                 # Dados adicionais (food classification)
└── outputs/              # Saídas geradas
```

## 🔧 Configurações Importantes

- **Device:** O código detecta automaticamente se há GPU disponível
- **Batch Size:** 64 (pode ser ajustado no código)
- **Learning Rate:** 0.0001
- **Epochs:** 100 (pode ser ajustado)
- **Tamanho da Imagem:** 32x32 pixels (ou 256x256 no GAN)

## 📊 Métricas de Avaliação

O projeto calcula as seguintes métricas:
- **ACC (Accuracy):** Acurácia geral
- **PRE (Precision):** Precisão
- **REV (Recall):** Revocação
- **F1:** Score F1

## ⚠️ Observações

- Certifique-se de que os arquivos `bird.zip` e `not-bird.zip` estão na raiz do projeto
- O treinamento pode demorar dependendo do hardware disponível
- Se não houver GPU, o treinamento será mais lento mas ainda funcionará

## 🎨 GAN para Geração de Imagens de Comida (`food_gan.py`)

O projeto também inclui um script Python standalone que implementa uma **DCGAN (Deep Convolutional Generative Adversarial Network)** para gerar imagens de comida.

### Pré-requisitos para `food_gan.py`

- Python 3.8 ou superior
- PyTorch e Torchvision instalados
- Pasta `data/food/` com imagens de comida (o script busca recursivamente)

### Instalação das Dependências

```bash
pip install torch torchvision pillow numpy
```

Ou use o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Como Executar o `food_gan.py`

O script possui dois modos de operação: **treinamento** e **geração de amostras**.

#### Modo 1: Treinamento (Train)

Treina a GAN com as imagens da pasta `data/food`:

**Comando básico:**
```bash
python food_gan.py --mode train --data_dir data/food --output_dir outputs/food_gan
```

**Comando simplificado (usa valores padrão):**
```bash
python food_gan.py
```

**Com parâmetros personalizados:**
```bash
python food_gan.py --mode train \
  --data_dir data/food \
  --output_dir outputs/food_gan \
  --epochs 50 \
  --batch_size 64 \
  --image_size 64 \
  --lr 0.0002
```

#### Modo 2: Gerar Amostras (Sample)

Gera imagens a partir de um checkpoint treinado:

```bash
python food_gan.py --mode sample \
  --checkpoint outputs/food_gan/generator_last.pt \
  --num_samples 16 \
  --output_dir outputs/food_gan
```

### Parâmetros do `food_gan.py`

#### Parâmetros Gerais

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--mode` | `train` ou `sample` | `train` | Modo de operação: treinamento ou geração |
| `--data_dir` | string | `data/food` | Diretório com imagens de treinamento |
| `--output_dir` | string | `outputs/food_gan` | Diretório para salvar checkpoints e amostras |
| `--device` | string | `cuda` ou `cpu` | Dispositivo a usar (detecta automaticamente) |

#### Parâmetros de Treinamento

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--epochs` | int | `30` | Número de épocas de treinamento |
| `--batch_size` | int | `128` | Tamanho do batch de treinamento |
| `--image_size` | int | `64` | Tamanho da imagem (quadrado) após redimensionamento |
| `--latent_dim` | int | `128` | Dimensão do vetor latente para entrada do gerador |
| `--lr` | float | `0.0002` | Learning rate para o otimizador Adam |
| `--beta1` | float | `0.5` | Beta1 para o otimizador Adam |
| `--beta2` | float | `0.999` | Beta2 para o otimizador Adam |
| `--save_every` | int | `400` | Salvar grade de amostras a cada N iterações |

#### Parâmetros de Geração

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--checkpoint` | string | `""` | Caminho para o checkpoint do gerador (obrigatório no modo sample) |
| `--num_samples` | int | `16` | Número de imagens a gerar no modo sample |

### Exemplos Práticos

#### 1. Treinamento Básico
```bash
python food_gan.py
```

#### 2. Treinamento Rápido para Teste
```bash
python food_gan.py --mode train --epochs 5 --batch_size 32
```

#### 3. Treinamento com Configuração Personalizada
```bash
python food_gan.py --mode train \
  --data_dir data/food \
  --output_dir outputs/food_gan \
  --epochs 100 \
  --batch_size 64 \
  --image_size 128 \
  --latent_dim 256 \
  --lr 0.0001 \
  --save_every 200
```

#### 4. Gerar 32 Imagens de Exemplo
```bash
python food_gan.py --mode sample \
  --checkpoint outputs/food_gan/generator_last.pt \
  --num_samples 32 \
  --output_dir outputs/food_gan
```

#### 5. Forçar Uso de CPU
```bash
python food_gan.py --mode train --device cpu
```

### Saídas do Treinamento

Durante o treinamento, o script salva automaticamente:

- **`generator_last.pt`** - Checkpoint do gerador (atualizado a cada época)
- **`discriminator_last.pt`** - Checkpoint do discriminador (atualizado a cada época)
- **`samples_eXXX_sXXXXXX.png`** - Amostras geradas durante o treinamento (a cada `--save_every` iterações)
- **`fixed_eXXX.png`** - Amostras geradas a partir de ruído fixo (salvo a cada época)

### Estrutura de Arquivos Esperada

```
CNN_Bird_NotBird/
├── food_gan.py              # Script principal
├── data/
│   └── food/                # Pasta com imagens de comida
│       ├── Bread/
│       ├── Dairy product/
│       ├── Dessert/
│       └── ...              # O script busca recursivamente
└── outputs/
    └── food_gan/            # Saídas do treinamento
        ├── generator_last.pt
        ├── discriminator_last.pt
        ├── samples_e*.png
        └── fixed_e*.png
```

### O que o Script Faz

1. **Carregamento de Dados:**
   - Busca recursivamente todas as imagens na pasta `data/food`
   - Suporta formatos: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`
   - Redimensiona e normaliza as imagens automaticamente

2. **Treinamento:**
   - Inicializa o Gerador e Discriminador
   - Treina ambos de forma adversária
   - Salva checkpoints e amostras periodicamente

3. **Geração:**
   - Carrega um checkpoint treinado
   - Gera imagens a partir de ruído aleatório
   - Salva as imagens geradas em uma grade

### Solução de Problemas para `food_gan.py`

**Erro: "No images found in ...":**
- Verifique se a pasta `data/food` existe e contém imagens
- O script busca recursivamente em todas as subpastas
- Certifique-se de que as imagens estão nos formatos suportados

**Erro de memória (Out of Memory):**
- Reduza o `--batch_size` (ex: 32 ou 64)
- Reduza o `--image_size` (ex: 32 ou 48)
- Feche outros programas que usam GPU

**Treinamento muito lento:**
- Verifique se a GPU está sendo usada: `--device cuda`
- Reduza o `--batch_size` se necessário
- Considere reduzir o `--image_size`

**Checkpoint não encontrado no modo sample:**
- Certifique-se de que o treinamento foi concluído
- Verifique o caminho do checkpoint: `--checkpoint outputs/food_gan/generator_last.pt`
- Use caminho absoluto se necessário

**Imagens geradas de baixa qualidade:**
- Treine por mais épocas: `--epochs 100` ou mais
- Aumente o `--latent_dim` (ex: 256)
- Use um `--image_size` maior (ex: 128)
- Verifique se há imagens suficientes na pasta de treinamento

## 🐛 Solução de Problemas (Notebooks)

**Erro ao carregar imagens:**
- Verifique se os arquivos ZIP estão no diretório correto
- Certifique-se de que os ZIPs contêm imagens válidas (.png, .jpg, .jpeg)

**Erro de memória:**
- Reduza o número de imagens carregadas (parâmetro `max` na função `loadImages`)
- Reduza o `batch_size` no DataLoader

**GPU não detectada:**
- Verifique se o PyTorch foi instalado com suporte CUDA
- O código funcionará em CPU, apenas será mais lento

