# ML Wardrobe Recommender Project

## 🎯 Project Overview

This project builds a fashion recommendation system that analyzes clothing images and recommends similar items based on visual features. The system uses deep learning (CNN) for feature extraction, followed by KNN for similarity matching and clustering for style classification.

---

## 🔬 CNN Feature Extraction Module

### Implementation

1. **Feature Extraction Pipeline**
   - Used **ResNet50** (pre-trained on ImageNet) for feature extraction
   - Processed **41,802 clothing images** from the Fashion Product Images Dataset
   - Extracted **2048-dimensional feature vectors** for each image
   
2. **Post-processing**
   - Applied **L2 normalization** for cosine similarity calculations
   - Reduced dimensionality using **PCA (512 components, 94.25% variance retained)**
   - Generated metadata mappings for all images

3. **Quality Validation**
   - Achieved **high similarity** (0.8-0.96) for same-category items
   - Validated features using t-SNE visualization and category consistency tests

---

## 📦 Extracted Features

Pre-extracted features are available in the `extracted_features/` folder:

### Available in repository:
- **`resnet50_features_pca512.npy`** - (41802, 512) PCA-reduced features
- **`resnet50_metadata.csv`** - Image metadata (ID, category, color, gender, etc.)
- **`resnet50_pca512_model.pkl`** - PCA model for new images
- **`resnet50_pca512_info.json`** - PCA statistics (94.25% variance retained)
- **`validation/`** - Feature quality validation results

### Large files (not in repo, contact team):
- **`resnet50_features_normalized.npy`** - (41802, 2048) Full L2-normalized features
- **`resnet50_features.npy`** - (41802, 2048) Original features

See `extracted_features/README.md` for details.

---

## 📊 Performance Metrics

- **Total Images Processed:** 41,802
- **Success Rate:** 99.99% (only 5 images failed)
- **Feature Dimension:** 2048 (original) / 512 (PCA)
- **Example Similarities:**
  - Shirts → Shirts: 0.96-1.0
  - Shoes → Shoes: 0.81-0.91
  - Sunglasses → Sunglasses: 0.95-0.96

---

## 🗂️ Project Structure

```
ML_Wardrobe_Recommender_Project/
│
├── CNN Feature Extraction:
│   ├── load_data.py                # Data loading from Kaggle
│   ├── clean_data.py               # Data cleaning
│   ├── eda.py                      # Exploratory data analysis
│   ├── image_loader.py             # Image preprocessing
│   ├── cnn_model.py                # ResNet50 feature extractor
│   ├── extract_all_features.py     # Batch feature extraction
│   ├── postprocess_features.py     # Normalization & PCA
│   ├── validate_features.py        # Quality validation
│   └── feature_extraction_api.py   # API for advanced usage
│
├── extracted_features/             # Pre-extracted features
│   ├── README.md                   # Feature files documentation
│   ├── resnet50_features_pca512.npy
│   ├── resnet50_metadata.csv
│   └── validation/                 # Quality validation results
│
├── eda_visuals/                    # EDA visualizations
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📋 Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Key packages:
# - tensorflow==2.13.0
# - scikit-learn==1.3.0
# - pandas==2.0.3
# - numpy==1.24.3
# - Pillow==10.0.0
```

---

## 🎓 Dataset

**Fashion Product Images Dataset** from Kaggle
- Source: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- Original images: ~44,000
- After cleaning: 41,802 images
- Categories: Apparel, Accessories, Footwear

---

## 🚀 Usage Example

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load PCA features and metadata
features = np.load('extracted_features/resnet50_features_pca512.npy')
metadata = pd.read_csv('extracted_features/resnet50_metadata.csv')

# Find similar items
query_idx = 0
query_vector = features[query_idx:query_idx+1]
similarities = cosine_similarity(query_vector, features)[0]
top_5 = np.argsort(similarities)[::-1][1:6]

# Show results
print(metadata.iloc[top_5][['id', 'articleType', 'baseColour']])
```

**For more examples, see code comments in each module.**