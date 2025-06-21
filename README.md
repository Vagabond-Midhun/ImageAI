# 🌸 ImageAI - Flower Species Classifier

This project uses deep learning and transfer learning to classify images of five flower types:
**daisy**, **dandelion**, **roses**, **sunflowers**, and **tulips**.

## 🔧 Setup
```bash
pip install -r requirements.txt
```

## 🏋️‍♂️ Training
Organize your dataset into this structure:
```
dataset/
├── train/
│   ├── daisy/
│   ├── dandelion/
│   ├── roses/
│   ├── sunflowers/
│   └── tulips/
└── val/
    ├── daisy/
    ├── dandelion/
    ├── roses/
    ├── sunflowers/
    └── tulips/
```
Then train your model:
```bash
python train.py
```

## 🚀 Run the Web App
```bash
streamlit run app.py
```

## ✅ Features
- Transfer learning with MobileNetV2
- Data augmentation (flip, rotate, zoom)
- Streamlit interface
- Real-time prediction with confidence score
- Preprocessing visualization
- Button to retrain model with new data
- Training performance graph