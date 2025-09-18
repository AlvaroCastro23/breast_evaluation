# Desafio Técnico


Sistema de Classificação de Tecido Mamário 🏥
Um sistema automatizado end-to-end para classificação de tecido mamário usando Deep Learning, desenvolvido para o Sistema Nacional de Saúde Português.
📋 Descrição
Este projeto implementa uma plataforma completa que permite aos médicos fazer upload de imagens de tecido mamário e receber um diagnóstico imediato através de um modelo de Rede Neural Convolucional (CNN) treinado no dataset BREASTMNIST.
🎯 Funcionalidades

Interface Web Intuitiva: Frontend moderno e responsivo para upload de imagens
Classificação Automatizada: Modelo CNN que classifica tecido como maligno ou não-maligno
Confiança do Diagnóstico: Exibe probabilidades e nível de confiança da classificação
Processamento em Tempo Real: Análise instantânea das imagens enviadas
Avisos de Segurança: Alertas para casos de baixa confiança

🚀 Como Executar
1. Instalar Dependências
bashpip install -r requirements.txt
2. Treinar o Modelo
bashpython train.py
Este comando irá:

Baixar automaticamente o dataset BREASTMNIST
Treinar o modelo CNN
Salvar o melhor modelo como best_model.pth

3. Executar a Aplicação Web
bashpython app.py
4. Acessar o Sistema
Abra seu navegador e acesse: http://localhost:8080
📁 Estrutura do Projeto
├── app.py              # Aplicação Flask principal
├── model.py            # Definição da arquitetura CNN
├── train.py            # Script de treinamento
├── utils.py            # Funções utilitárias
├── requirements.txt    # Dependências do projeto
├── templates/
│   └── index.html      # Interface web
└── best_model.pth      # Modelo treinado (gerado após treinamento)
🏗️ Arquitetura do Modelo
O modelo SmallCNN implementa uma arquitetura simples mas eficaz:

3 Camadas Convolucionais com BatchNorm e MaxPooling
2 Camadas Fully Connected com Dropout
Ativação ReLU e Softmax para classificação binária
Input: Imagens 64x64 em escala de cinza
Output: Probabilidades para classes maligno/não-maligno

📊 Dataset

BREASTMNIST: Dataset médico padronizado
Classes: Maligno (1) vs Não-Maligno (0)
Formato: Imagens 28x28 redimensionadas para 64x64
Divisão: Train/Validation/Test automática

⚡ Tecnologias Utilizadas

Backend: Flask, PyTorch, PIL
Frontend: HTML5, CSS3, JavaScript
ML: CNN personalizada, MedMNIST
Deploy: Python WSGI compatível

🛡️ Considerações de Segurança

⚠️ Uso Educacional: Esta ferramenta é apenas para fins educacionais
🩺 Consulta Médica: Sempre consulte um profissional médico para diagnósticos definitivos
🔍 Confiança: O sistema alerta quando a confiança está baixa (<70%)

📈 Melhorias Futuras

 Implementar data augmentation
 Adicionar interpretabilidade (Grad-CAM)
 Otimização de hiperparâmetros
 Validação cruzada
 Métricas médicas específicas (AUC, Sensibilidade, Especificidade)
 Sistema de logging e monitoramento

👨‍💻 Desenvolvedor
Projeto desenvolvido como parte do desafio técnico para sistemas de saúde automatizados.
🔧 Resolução de Problemas
Erro: "cannot import name 'SmallCNN'"

Certifique-se de que o arquivo model.py contém a definição da classe
Verifique se não há erros de sintaxe no arquivo

Erro: "No module named 'medmnist'"
bashpip install medmnist
Erro: "CUDA out of memory"

O modelo funciona perfeitamente em CPU
Reduza o batch_size se necessário

Problema: Baixa acurácia

Execute mais épocas de treinamento
Verifique se o dataset foi baixado corretamente

📝 Licença
Este projeto é de código aberto e está disponível sob a licença MIT.
