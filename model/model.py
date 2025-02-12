from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


class Model:
    def __init__(self):
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        classifier = LogisticRegression(max_iter=500)
        self.model = make_pipeline(vectorizer, classifier)
        self.label_encoder = LabelEncoder()

    def train(self, x, y):
        y_encoded = self.label_encoder.fit_transform(y)
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y_encoded, test_size=0.2)
        print('Training model...')
        self.model.fit(x_train, y_train)
        print(f"Model Accuracy: {self.model.score(x_test, y_test) * 100:.2f}%")

    def predict(self, text):
        pred = self.model.predict([text])
        return self.label_encoder.inverse_transform(pred)[0]
