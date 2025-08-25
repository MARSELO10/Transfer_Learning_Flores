...............![flores](https://github.com/user-attachments/assets/01e86e8e-2f6b-4a64-853e-d2123ec83f7a)...............

---

# 🌹🌻Transfer Learning - Classificação - Flores

[![Python](https://img.shields.io/badge/Python-3.12.6-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

---

## 📋 - Descrição 
Este projeto é uma tarefa dada aos estudantes do BairesDev - Machine Learning Training, um bootcamp da plataforma [DIO](https://www.dio.me/).

---

## 🎯 - Sobre o Projeto de Redução de Dimensionalidade
Este projeto implementa um classificador de imagens usando Transfer Learning para distinguir entre 5 tipos de flores.

Cinco categorias de imagens de flores estão sendo treinadas por uma CNN, com base no ResNet e por meio de aprendizagem por transferência, para fornecer um classificador de categorias de flores. As flores podem estar nas seguintes categorias: margarida, dente-de-leão, rosas, girassóis ou tulipas.

---

## 🚀 Funcionalidades

* ✅ Download automático do dataset Oxford Flowers 102
* ✅ Extração e organização das imagens + rótulos (.mat)
* ✅ Carregamento de rótulos e divisões do dataset (treino, validação e teste) via scipy.io
* ✅ Classe personalizada FlowersDataset para integração com PyTorch DataLoader
* ✅ Pré-processamento com transforms: redimensionamento para 224x224, conversão para tensor e normalização (ImageNet mean/std)
* ✅ Suporte a DataLoaders com batch size e embaralhamento configuráveis
* ✅ Uso do ResNet50 pré-treinado (ImageNet) com ajuste da última camada para 102 classes
* ✅ Treinamento com CrossEntropyLoss e Adam Optimizer
* ✅ Função de validação com cálculo de acurácia em cada época
* ✅ Loop de treinamento com barra de progresso (tqdm) e monitoramento da perda
* ✅ Função de avaliação final no conjunto de teste com relatório da acurácia

---

## 🛠️ Tecnologias Utilizadas

* Python 3
* PyTorch (torch, torch.nn, torch.utils.data, torchvision)
* scipy.io (para leitura de arquivos .mat)
* scikit-learn (métricas de avaliação – accuracy_score)
* tqdm (barra de progresso no treinamento)
* PIL (Pillow) (manipulação de imagens)
* Google Colab (ambiente de execução)
* wget / tar / os (download e manipulação de arquivos/diretórios)

---

## 📁 Instrução de Uso Google Collab

1. **Todas instruções está documentada no próprio arquivo do Collab**

2. **Execução passo a passo: Execute as células em ordem numérica (1 → 11).**
   
   **O dataset será baixado e organizado automaticamente na pasta /content/Dataset_Flowers.**
   
   **Os rótulos e divisões (treino, validação e teste) serão carregados via arquivos .mat.**
   
   **Os DataLoaders e o modelo ResNet50 pré-treinado serão configurados automaticamente.**

3. **Reutilizar o modelo sem treinar novamente:**

   **Ao final do treinamento, salve o modelo treinado:**

   **torch.save(model.state_dict(), "flowers_resnet50.pth")**

   **Para carregar novamente sem treinar:**

   **model.load_state_dict(torch.load("flowers_resnet50.pth"))**

   **model.eval()**

---

## 📁 Estrutura do Projeto

```

projeto
oxford-flowers102.py
├── FlowersDataset (class)        # Classe customizada para carregar imagens e rótulos
│   ├── __init__()                # Inicializa diretório, índices, labels e transformações
│   ├── __len__()                 # Retorna tamanho do dataset
│   └── __getitem__()             # Retorna imagem transformada + rótulo correspondente
│
├── validate_model()              # Avalia modelo no conjunto de validação
├── train_model()                 # Loop de treino com barra de progresso (tqdm)
├── test_model()                  # Avalia modelo final no conjunto de teste
└── main (sequência do script)    # Download, extração, dataloaders, treinamento e teste

```

---

## 📊 Aplicações em Machine Learning

* Classificação de imagens em múltiplas classes (102 tipos de flores)

* Transfer Learning com modelos pré-treinados (ResNet50 no ImageNet)

* Extração de características visuais (features aprendidas pela CNN)

* Treinamento supervisionado com rótulos fornecidos

* Avaliação de desempenho (acurácia em treino, validação e teste)

* Fine-tuning de CNNs para domínios específicos (flores → botânica, biologia, agricultura)

---

## 🤝 Patrocinadores ou Afins

Estes são os *links* das empresas/projetos ligados a este trabalho:

- [DIO](https://www.dio.me/): plataforma de cursos e bootcamps online.
- [Baires Dev](https://www.bairesdev.com/): A BairesDev é uma empresa multinacional americana de outsourcing de TI e desenvolvimento de software
- [University of Oxford – Visual Geometry Group (VGG)](https://www.robots.ox.ac.uk/~vgg/): Responsável pelo dataset Oxford Flowers 102
- [ImageNet](https://www.image-net.org/): Conjunto de dados no qual o ResNet50 foi pré-treinado (e cuja média/desvio padrão são usados na normalização)
- [PyTorch](https://pytorch.org/): Framework utilizado para desenvolvimento e treinamento do modelo
- [Google Collab](https://colab.google/): O Google Colab (ou Colaboratory) é um ambiente de desenvolvimento integrado (IDE) gratuito e baseado em nuvem que permite escrever e executar código Python diretamente no navegador, sem necessidade de instalação ou configuração.

---

# Contribuição

Contribuições são bem-vindas!\
Se você encontrar algum problema ou tiver sugestões de melhorias, por favor abra uma **issue** ou envie um **pull request** para o repositório.\
Ao contribuir para este projeto, siga as convenções de commits e envie suas alterações em um **branch** ou **file** separado.\
Saiba mais sobre o código de conduta acessando o link: [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

---

# Licença

Este repositório possui licença [MIT](https://github.com/MARSELO10/Transfer_Learning_Flores)
