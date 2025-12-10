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

## 🐛 Solução de Problemas

**Erro ao carregar imagens:**
- Verifique se os arquivos ZIP estão no diretório correto
- Certifique-se de que os ZIPs contêm imagens válidas (.png, .jpg, .jpeg)

**Erro de memória:**
- Reduza o número de imagens carregadas (parâmetro `max` na função `loadImages`)
- Reduza o `batch_size` no DataLoader

**GPU não detectada:**
- Verifique se o PyTorch foi instalado com suporte CUDA
- O código funcionará em CPU, apenas será mais lento

