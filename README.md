# ML Wardrobe Recommender Project

## 🎯 Project Overview

This project builds a fashion recommendation system that analyzes clothing images and recommends similar items based on visual features. The system uses deep learning (CNN) for feature extraction, followed by KNN for similarity matching and clustering for style classification.

---

## 👤 My Part: CNN Feature Extraction

**Responsible for:** Extracting high-dimensional feature vectors from clothing images using pre-trained CNNs.

### What Was Implemented

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

## 📦 Output Files for Team (in `extracted_features/`)

### Ready-to-use files:
- **`resnet50_features_normalized.npy`** - (41802, 2048) L2-normalized vectors for cosine similarity
- **`resnet50_features_pca512.npy`** - (41802, 512) PCA-reduced vectors for faster computation
- **`resnet50_metadata.csv`** - Image metadata (ID, category, color, gender, etc.)
- **`resnet50_extraction_info.json`** - Extraction statistics
- **`resnet50_pca512_info.json`** - PCA information (variance explained, etc.)

---

## 🚀 Quick Start for Team Members

### 1. Load the Features

```python
import numpy as np
import pandas as pd
import json

# Load feature vectors
features = np.load('extracted_features/resnet50_features_normalized.npy')
# OR for faster processing:
# features = np.load('extracted_features/resnet50_features_pca512.npy')

# Load metadata
metadata = pd.read_csv('extracted_features/resnet50_metadata.csv')

print(f"Features shape: {features.shape}")  # (41802, 2048) or (41802, 512)
print(f"Total images: {len(metadata)}")
```

### 2. KNN - Find Similar Items

```python
from sklearn.metrics.pairwise import cosine_similarity

# Query image (by index)
query_idx = 0
query_vector = features[query_idx:query_idx+1]

# Calculate similarity with all images
similarities = cosine_similarity(query_vector, features)[0]

# Get top-5 most similar (excluding query itself)
top_k = 5
top_indices = np.argsort(similarities)[::-1][1:top_k+1]

# Get recommendations
recommendations = metadata.iloc[top_indices]
print(recommendations[['id', 'articleType', 'baseColour']])
print(f"Similarities: {similarities[top_indices]}")
```

### 3. Clustering - Group by Style

```python
from sklearn.cluster import KMeans

# Use PCA version for speed
features_pca = np.load('extracted_features/resnet50_features_pca512.npy')

# K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(features_pca)

# Add cluster labels to metadata
metadata['cluster'] = clusters

# Analyze clusters
for i in range(10):
    cluster_items = metadata[metadata['cluster'] == i]
    print(f"\nCluster {i}: {len(cluster_items)} items")
    print(cluster_items['articleType'].value_counts().head(3))
```

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
├── CNN Module (My Implementation):
│   ├── load_data.py                # Data loading from Kaggle
│   ├── clean_data.py               # Data cleaning
│   ├── image_loader.py             # Image preprocessing
│   ├── cnn_model.py                # ResNet50 feature extractor
│   ├── extract_all_features.py     # Batch feature extraction
│   ├── postprocess_features.py     # Normalization & PCA
│   ├── validate_features.py        # Quality validation
│   └── feature_extraction_api.py   # API for team (optional)
│
├── eda_visuals/                    # Exploratory data analysis
├── dataset_config.json             # Dataset paths
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🔧 Using the API (Optional)

For advanced usage, you can use the provided API:

```python
from feature_extraction_api import FeatureExtractionAPI

# Initialize API
api = FeatureExtractionAPI()

# Get features for specific IDs
features = api.get_features_by_ids([15970, 39386, 59263])

# Get all items in a category
shirt_features, shirt_ids, shirt_meta = api.get_features_by_category('Shirts')

# Extract features from a new image
new_features = api.extract_features_from_new_image(
    image_id=12345,
    apply_normalization=True
)
```

---

## 💡 Recommendations for Team

### For KNN Implementation:
- Use `resnet50_features_normalized.npy` with cosine similarity
- Metric: `sklearn.metrics.pairwise.cosine_similarity`
- Expected precision@5: ~70-80% for same category

### For Clustering:
- Use `resnet50_features_pca512.npy` for faster computation
- Try K-Means, Hierarchical, or DBSCAN
- Use `articleType` or `masterCategory` for validation

### For Visualization:
- Apply t-SNE or UMAP on PCA features
- Color by category to validate clustering quality
- See examples in `extracted_features/validation/`

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

## ✅ Next Steps for Team

1. **KNN Team:** Implement similarity search using `extracted_features/resnet50_features_normalized.npy`
2. **Clustering Team:** Perform style classification using `extracted_features/resnet50_features_pca512.npy`
3. **Integration:** Combine CNN features with KNN and clustering for final recommendation system

---

## 📁 Quick File Reference

| File | Size | Purpose |
|------|------|---------|
| `resnet50_features_normalized.npy` | 327 MB | **Best for KNN** - L2 normalized, use with cosine similarity |
| `resnet50_features_pca512.npy` | 82 MB | **Best for Clustering** - Fast, 94% variance retained |
| `resnet50_metadata.csv` | 4 MB | Image metadata - links features to categories |
| `resnet50_pca512_model.pkl` | 4 MB | PCA model - for transforming new images |

**Good luck! 🚀**