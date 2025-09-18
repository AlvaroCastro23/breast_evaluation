
import io
import torch
from PIL import Image
from flask import Flask, request, render_template, jsonify
from model import SmallCNN
import torchvision.transforms as transforms

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Carregar modelo treinado ===
model = SmallCNN(in_channels=1, num_classes=2)
model.load_state_dict(torch.load('final_model.pth', map_location=device))
model.to(device)
model.eval()

# === Transformação consistente com treino ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # converte para 1 canal
    transforms.Resize((64,64)),                   # redimensiona para 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'no file'}), 400
    file = request.files['file']
    try:
        # Abrir imagem e converter para RGB (caso venha em outro modo)
        img = Image.open(file.stream).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device).float()

        # Inferência
        with torch.no_grad():
            out = model(img_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            prob = float(probs[pred])

        label_map = {0: 'non-malignant', 1: 'malignant'}
        return jsonify({'label': label_map[pred], 'probability': prob})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
