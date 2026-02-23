# OpenPlant ID (Free & Open-Source Plant Identifier)

OpenPlant ID is a production-oriented blueprint and implementation for a **completely free** plant identifier app (mobile + web + backend + training pipeline) using open-source models/datasets and no paid APIs.

## 1) Architecture Diagram

```mermaid
flowchart TD
    A[Mobile/Web App (Expo React Native)] -->|Upload image| B[FastAPI Inference API]
    B --> C[Preprocessing Pipeline]
    C --> D[Segmentation + Leaf foreground mask (OpenCV)]
    D --> E[ONNX Plant Classifier (EfficientNetV2/ViT/ConvNeXt/MobileNetV3)]
    D --> F[ONNX Disease Detector]
    E --> G[Plant Metadata + Care Knowledge Base]
    F --> G
    G --> A

    H[ML Training Pipeline] --> I[Dataset Index Builder]
    I --> J[PyTorch Fine-tuning]
    J --> K[Optimization (Pruning + Quantization)]
    K --> L[Export ONNX/TFLite/CoreML]
    L --> B
    L --> A
```

## 2) Folder Structure

```text
.
├── apps/mobile/                 # Cross-platform app (Android/iOS/Web)
├── services/api/                # FastAPI backend for online inference
├── ml/
│   ├── configs/                 # Train configs
│   ├── scripts/                 # Data prep, train, export, optimize
│   └── src/                     # Dataset & ML utility modules
└── deployment/                  # Docker compose for serving API
```

## 3) Full Source Code Overview

### Mobile app (`apps/mobile`)
- Camera/gallery image intake via `expo-image-picker`.
- API integration for identification.
- Result view includes:
  - common/scientific name
  - confidence score
  - description
  - care plan (watering/sunlight/soil/temp)
  - disease diagnosis/treatment
- Local history cache using Zustand.

### Backend API (`services/api`)
- Endpoints:
  - `GET /v1/health`
  - `POST /v1/identify` (multipart image upload)
- Pipeline:
  - image decode + normalization
  - OpenCV foreground masking for leaf-centric segmentation
  - ONNX Runtime inference
  - disease diagnosis inference
  - structured response with care metadata

### AI/ML (`ml`)
- Dataset unifier script: convert class folders into CSV index.
- Train script supports model options:
  - `efficientnet_v2_s`
  - `vit_b_16`
  - `convnext_tiny`
  - `mobilenet_v3_large`
- Export script for ONNX.
- Optimization script for pruning + dynamic quantization.

## 4) Model Training Pipeline

1. Prepare datasets (PlantNet/iNaturalist/LeafSnap/Kaggle folder exports):
   ```bash
   python ml/scripts/prepare_datasets.py --dataset-root data/plantnet --out data/index.csv
   ```
2. Train model:
   ```bash
   python ml/scripts/train.py --index data/index.csv --model efficientnet_v2_s --epochs 40 --batch-size 64
   ```
3. Export ONNX:
   ```bash
   python ml/scripts/export_models.py --weights artifacts/best.pt --num-classes <N>
   ```
4. Optimize for mobile:
   ```bash
   python ml/scripts/optimize.py --weights artifacts/best.pt --num-classes <N>
   ```

### Accuracy target strategy (90-98% top-1)
- Use transfer learning with progressive resizing (224->320).
- Merge datasets + label harmonization.
- Hard-example mining and class-balanced sampling.
- Ensemble server mode (ConvNeXt + ViT) with MobileNetV3 offline fallback.

## 5) Mobile/Web Implementation

```bash
cd apps/mobile
npm install
npx expo start            # native
npx expo start --web      # web
```

Set API url:
```bash
export EXPO_PUBLIC_API_BASE=http://localhost:8000/v1
```

## 6) Backend Deployment Instructions

### Local
```bash
cd services/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
cd deployment
docker compose up --build
```

## 7) Offline Model Conversion Guide

### ONNX -> TFLite
- Use `onnx-tf` to generate TensorFlow SavedModel.
- Use `tensorflow/lite` converter with int8 quantization.

### ONNX -> CoreML
- Use `coremltools.convert("model.onnx")`.

### Mobile integration
- Android: TFLite via `org.tensorflow:tensorflow-lite`.
- iOS: CoreML via Vision framework.
- Expo/React Native: bridge with native module or use server fallback.

## 8) Optimization Guide

- Post-training dynamic quantization on linear layers.
- Structured pruning on conv filters.
- Distillation from ViT/ConvNeXt teacher to MobileNetV3 student.
- Mixed precision inference on GPU where available.
- Keep mobile artifact under 100MB using int8 + distilled architecture.

## 9) Included Product Features

- ✅ Plant identification by photo
- ✅ Scientific/common names
- ✅ Confidence scores
- ✅ Plant descriptions
- ✅ Care instructions
- ✅ Watering + sunlight guidance
- ✅ Disease detection + health diagnosis
- ✅ History saving
- ✅ Fast API response structure (sub-second feasible with optimized ONNX)
- ✅ Offline-ready model export pipeline

## 10) Bangladesh Dataset Support (Bonus)

Add a `data/bd_plants/` folder in class-per-directory format and include it in your merged index generation script. Fine-tune final epochs with regional sampling weights to improve local plant recall.

---

This repo is intentionally free/open-source only, with all pipelines designed around self-hosted and local inference workflows.
