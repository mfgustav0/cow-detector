# IMC Bovino

Sistema de Visão Computacional para Estimativa de Peso, Altura, Comprimento e Índice de Massa Corporal (IMC) Bovino a partir de imagens e vídeos.

## Sobre o Projeto

O  IMC Bovino é um protótipo desenvolvido para aplicar técnicas de Visão Computacional na análise de bovinos. O sistema utiliza detecção automática de animais para estimar medidas corporais e gerar indicadores que auxiliam na avaliação visual do estado corporal do rebanho.

A proposta é oferecer uma ferramenta simples para análise preliminar, permitindo a obtenção de estimativas sem a necessidade de equipamentos físicos de medição.

## Funcionalidades

* Detecção automática de bovinos em imagens e vídeos.
* Estimativa de peso corporal.
* Estimativa de altura da cernelha.
* Estimativa de comprimento corporal.
* Cálculo do IMC Bovino.
* Geração de Score de Saúde.
* Emissão de alertas automáticos.
* Interface gráfica para operação do sistema.
* Geração de relatório em PDF contendo os resultados da análise.
* Processamento contínuo de vídeos com captura de amostras para documentação.

## Fluxo de Funcionamento

1. O usuário seleciona uma imagem ou vídeo.
2. O sistema detecta automaticamente os bovinos presentes.
3. São calculadas métricas corporais estimadas:
   * Peso
   * Altura
   * Comprimento
   * Robustez corporal
   * IMC Bovino
4. É gerado um score de saúde baseado nas métricas obtidas.
5. O sistema apresenta alertas quando identifica possíveis condições fora dos parâmetros esperados.
6. Ao finalizar a análise, pode ser gerado um relatório em PDF.

## Tecnologias Utilizadas

* Python
* OpenCV
* NumPy
* Tkinter
* Pillow
* Matplotlib
* ReportLab
* YOLO (Ultralytics)

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

## Como Utilizar

### Imagem

1. Clique em **Abrir Imagem**.
2. Selecione uma fotografia do animal.
3. Aguarde o processamento.
4. Os resultados serão exibidos na tela.

### Vídeo

1. Clique em **Abrir Vídeo**.
2. Selecione o arquivo desejado.
3. O sistema realizará análises periódicas durante a reprodução.
4. Ao finalizar, clique em **Parar e Gerar Relatório** para exportar os resultados.

## Recomendações para Captura

Para obter melhores resultados:

* Fotografar ou filmar o animal lateralmente.
* Manter boa iluminação.
* Evitar obstruções no corpo do animal.
* Utilizar imagens com boa resolução.
* Manter o animal ocupando parte significativa da cena.

## Limitações

* As medidas apresentadas são estimativas computacionais.
* O sistema não substitui medições zootécnicas realizadas em campo.
* O score de saúde possui caráter indicativo e não constitui diagnóstico veterinário.
* A precisão depende da qualidade da imagem, posição do animal e desempenho do modelo de detecção.
* Diferentes raças e faixas etárias podem apresentar variações que afetam as estimativas.

## Aplicações

* Monitoramento preliminar do estado corporal.
* Apoio à gestão de rebanhos.
* Estudos acadêmicos em Visão Computacional.
* Demonstrações educacionais e experimentais.
* Projetos de pesquisa em agropecuária digital.

## Aviso

Este sistema possui finalidade acadêmica e experimental. Os resultados devem ser utilizados apenas como apoio à análise visual e não substituem avaliações realizadas por médicos veterinários ou profissionais da área zootécnica.
