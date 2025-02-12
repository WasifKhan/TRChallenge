# Legal Text Classification Model

## 📚 Overview
This project aims to classify **legal text paragraphs** into **categorical labels** based on key phrases present in the document. The model processes legal documents and predicts categories such as **"On Appeal"** and **"Appellate Review"**.

The approach includes:
- **Text preprocessing** (cleaning, tokenization)
- **Feature extraction** using **TF-IDF**
- **Label encoding** for categorical variables
- **Supervised classification** (Logistic Regression, or fine-tuned SBERT)

---

## Results
- **Number of Documents: 18000**
- **Number of Postures: 27659**
- **Number of Paragraphs: 542169**
- **Model Accuracy: 71%**


## 📂 Directory Structure
```
/legal-text-classification
🗂️ main.py                # Main script to load data, train, and test the model
🗁️ model/
   └️ model.py           # Defines the classification model (TF-IDF/SBERT)
🗁️ data/
   └️ data.py            # Data processing, tokenization, and label encoding
📚 TRDataChallenge2023.zip # ZIP file **NEEDS TO BE MANUALLY INCLUDED IN DIRECTORY**
📁 requirements.txt        # Dependencies for the project
📄 README.txt               # Documentation
```

---

## 📃 Data Processing

### 💡 Input Data Format
The dataset is stored in **`TRDataChallenge2023.zip`**, which contains one JSON file. The file follows the structure of the **first JSON object** and is formatted into a dataset described in the **second and third JSON object**
```json
{'documentId:': '14342',
 'postures': ['On Appeal', 'Summary'],
 'sections': [{'headtext': "Background",
               'paragraphs': ['paragraph 1', 'paragraph 2']}
             ]
}

{
  "x": "Background:\nparagraph 1\n aragraph 2",
  "y": "On Appeal"
}
and
{
  "x": "Background:\nparagraph 1\n aragraph 2",
  "y": "Summary"
}

```

- **`x`**: Headtext + all paragraphs from legal documents
- **`y`**: The **categorical label**, described with keywords

### 💡 Data Loading (data.py)
- Extracts **JSON files** from the ZIP archive (*extract_zip*)
- Handles nested JSON structures, combining multiple paragraphs (*prepare_data*)
- Provides output as per requirements in challenge (*describe*)
- Handles cleanup, removing temporary files (*clean*)

---

## 🌟 Model Implementation (model.py)
### 1️⃣ Tokenization & Vectorization (TF-IDF)
- Converts legal text into **numerical features** using **TF-IDF Vectorization**.
- Captures the importance of words in the text relative to the entire dataset.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_transformed = tfidf_vectorizer.fit_transform(X)
```

### 2️⃣ Label Encoding
- Converts categorical labels (`y`) into numerical format using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```
- Example Mapping:
  ```
  'On Appeal' → 0
  'Appellate Review' → 1
  ```

---

## Model Implementation
### **Pipeline: TF-IDF + Logistic Regression**
To streamline the process, a **Scikit-Learn pipeline** integrates the **TF-IDF Vectorizer** with **Logistic Regression**.

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(max_iter=500)
model = make_pipeline(tfidf_vectorizer, classifier)
```

### **Training the Model**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
```

### **Evaluating the Model**
```python
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

### **Improvement: SBERT + SVM (Sentence Transformers)**
- Uses **pretrained `all-MiniLM-L6-v2` model** for embeddings.
- Trains an **SVM classifier** on vectorized text.
- I would have done this but laptop doesn't have a GPU :(

```python
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
X_train_embed = sbert_model.encode(X_train)
svm_clf = SVC(kernel="linear").fit(X_train_embed, y_train)
```

---

## 🚀 Running the Model

### **1️⃣ Install Dependencies**
Ensure you have Python 3 installed. Then, run:
```bash
pip install -r requirements.txt
```

### **2️⃣ Extract Data**
Make sure `TRDataChallenge2023.zip` is in the project directory.

### **3️⃣ Run the Model**
Execute the main script to **train and evaluate the model**:
```bash
python main.py
```

### **4️⃣ Predict a Category**
You can test a new legal paragraph using:
```python
from model.model import predict

text = "The defendant filed an appeal against the trial court decision..."
print(predict(text))
```

---

## 📊 Expected Output
After training, the model predicts the category **y** based on legal text **x**:
```
Input: "After pleading guilty, Howard appeals his 168-month sentence..."
Predicted Category: "Appellate Review"
```

---

## 📚 Dependencies (`requirements.txt`)
```txt
numpy
pandas
scikit-learn
nltk
transformers
sentence-transformers
torch
matplotlib
re
zipfile
json
pathlib
```

---

## 🛠 Future Improvements
- **Use SBERT Embeddings (Sentence Transformers) Archiecture:** Captures sentence semantics.
- **Fine-tuning BERT on legal text** for better accuracy.
- **Use Named Entity Recognition (NER)** to extract important case details.
- **Train on more legal documents** to generalize predictions.

---

```


