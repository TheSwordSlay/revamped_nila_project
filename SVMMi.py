from nltk.corpus import stopwords
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from functools import partial

def SVMmi(df):
    with open('svm_model_mi.pkl', 'rb') as file:
        svm_selected = pickle.load(file)

    mi_selector = SelectKBest(
        score_func=partial(mutual_info_classif, random_state=42),
        k=200  # Select top 5 features
    )

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def preprocess_indonesian_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Stemming
        text = stemmer.stem(text)
        # Remove Indonesian stopwords
        stop_words = set(stopwords.words('indonesian'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    scaler = StandardScaler()

    test = df
    train = pd.read_csv('modified_data.csv')

    X_train_scaled = model.encode(train['cleaned_text'].tolist(), show_progress_bar=True)
    X_train_scaled = scaler.fit_transform(X_train_scaled)
    y_train = train['skor']

    X_train_scaled = mi_selector.fit_transform(X_train_scaled, y_train)

    test['cleaned_text'] = test['Content'].apply(preprocess_indonesian_text)
    X_test = test['cleaned_text']
    y_test = test['Skor']

    X_test_sentencetransformer = model.encode(X_test.tolist(), show_progress_bar=True)
    X_test_sentencetransformer = scaler.fit_transform(X_test_sentencetransformer)
    X_test_sentencetransformer = mi_selector.transform(X_test_sentencetransformer)
    y_pred = svm_selected.predict(X_test_sentencetransformer)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Akurasi Model: {accuracy * 100:.2f}%")
    results_df = pd.DataFrame({
        'preprocessed text': X_test,
        'sentiment': y_test,
        'prediction': y_pred
    })
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_report': class_report
    }
    
    return results_df, metrics