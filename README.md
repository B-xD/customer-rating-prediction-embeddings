# Customer Rating Prediction from Text Reviews

This repository demonstrates an end-to-end pipeline for predicting customer ratings based on textual comments/reviews using **Hugging Face sentence embeddings** and a **linear regression model**.

The project focuses on transforming unstructured text into numerical representations (embeddings) and applying a simple, interpretable machine learning model to predict customer ratings.

---

## 🚀 Project Overview

Customer reviews contain rich semantic information that can be leveraged to predict user satisfaction. In this project:

1. Customer comments are converted into dense vector embeddings using a **pretrained Hugging Face model**.
2. Embeddings are aggregated using **mean pooling**.
3. A **Linear Regression** model is trained to predict customer ratings from these embeddings.

This approach provides a lightweight baseline for text-to-score prediction tasks.

---

## 🧠 Model & Embeddings

### Embedding Model

* **Model**: `BAAI/bge-m3`
* **Source**: Hugging Face Transformers
* **Pipeline**: `feature-extraction`

The `bge-m3` model is a multilingual, high-performance embedding model suitable for semantic similarity and downstream prediction tasks.

### Pooling Strategy

The output of the embedding model is a sequence of token vectors. These are aggregated using **mean pooling**:

```python
np.mean(token_embeddings, axis=0)
```

This produces a single fixed-length vector per review.

---

## 📊 Machine Learning Model

* **Algorithm**: Linear Regression
* **Library**: scikit-learn

Linear Regression is used as a baseline model to:

* Keep the solution interpretable
* Evaluate how well embeddings alone capture rating-related information

---

## 🧪 Workflow

### 1. Load Reviews

Customer comments are extracted from the dataset:

```python
strings = train_and_test["comment"].tolist()
```

### 2. Generate Embeddings

Each comment is embedded using the Hugging Face pipeline:

```python
pipe = pipeline('feature-extraction', model="BAAI/bge-m3", device=0)
embedding_strings = [custom_func(x) for x in pipe(strings)]
```

### 3. Create Feature Matrix

Embeddings are stored in a Pandas DataFrame:

```python
emb_dataset = pd.DataFrame(embedding_strings)
```

### 4. Train Regression Model

```python
from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(emb_dataset, ratings)
```

### 5. Predict Ratings

```python
predictions = reg_model.predict(emb_dataset)
```

---

## 📦 Dependencies

* Python 3.9+
* transformers
* torch
* numpy
* pandas
* scikit-learn
* tqdm

Install dependencies with:

```bash
pip install transformers torch numpy pandas scikit-learn tqdm
```

---

## 🔍 Use Cases

* Customer satisfaction analysis
* Review scoring automation
* Sentiment-aware rating prediction
* NLP feature engineering baseline

---

## ⚠️ Notes & Limitations

* Linear Regression is used for simplicity; performance can be improved with:

  * Regularized models (Ridge, Lasso)
  * Tree-based models
  * Neural networks
* No fine-tuning is applied to the embedding model
* Mean pooling may lose token-level nuance

---

## 📌 Future Improvements

* Train/test split and proper evaluation metrics (RMSE, MAE)
* Hyperparameter tuning
* Model comparison
* Dimensionality reduction (PCA)
* Fine-tuned transformer models

---

## 📄 License

This project is provided for educational and research purposes.

---

If you find this project useful, feel free to ⭐ the repository!
