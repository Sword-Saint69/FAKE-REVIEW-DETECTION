import pandas as pd
import numpy as np
import pickle
import os
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

# Models directory create cheyyuka, illenkil
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# CPU coresinte count set cheyyuka (system processesinu 1 core free vechuka)
n_jobs = max(1, multiprocessing.cpu_count() - 1)

def print_progress(message):
    """Timestamp upayogichu progress message print cheyyuka"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def load_dataset(file_path='fake_reviews_dataset.csv', sample_size=None):
    """Dataset load cheyyuka, preprocess cheyyuka, optional sampling upayogikkuka"""
    try:
        print_progress("Dataset load cheyyunnu...")
        # Dataset load cheyyuka
        df = pd.read_csv(file_path)
        
        # Faster traininginu vendi optional subset sample cheyyuka
        if sample_size and sample_size < len(df):
            # Class distribution maintain cheyyan stratified sampling
            genuine = df[df['label'] == 0].sample(n=min(sample_size//2, len(df[df['label'] == 0])), random_state=42)
            fake = df[df['label'] == 1].sample(n=min(sample_size//2, len(df[df['label'] == 1])), random_state=42)
            df = pd.concat([genuine, fake]).sample(frac=1, random_state=42).reset_index(drop=True)
            print_progress(f"Faster traininginu vendi {len(df)} reviewsinte sample upayogikkunnu")
        
        print_progress(f"Dataset load cheythu {len(df)} reviews")
        print(f"Genuine reviews: {len(df[df['label'] == 0])}")
        print(f"Fake reviews: {len(df[df['label'] == 1])}")
        return df
    except Exception as e:
        print_progress(f"Dataset load cheyyumbol error: {str(e)}")
        return None

def train_and_evaluate_model(name, model_class, X_train, X_test, y_train, y_test, params=None):
    """Oru single model train cheyyuka, evaluate cheyyuka"""
    start_time = time.time()
    
    # Parameters provide cheythal model initialize cheyyuka
    if params:
        model = model_class(**params)
    else:
        model = model_class()
    
    # Model train cheyyuka
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Trained model save cheyyuka
    with open(f"{MODEL_DIR}/{name}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Model evaluate cheyyuka
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    return {
        'name': name,
        'accuracy': float(accuracy),
        'report': report,
        'confusion_matrix': conf_matrix,
        'training_time': training_time
    }

def train_models(X_train, X_test, y_train, y_test):
    """Ella modelsum parallel aayi train cheyyuka"""
    # Optimized parameters upayogichu models define cheyyuka
    model_configs = [
        ('naive_bayes', MultinomialNB, {'alpha': 0.1}),
        ('linear_svc', LinearSVC, {'C': 1.0, 'max_iter': 1000, 'dual': False}),
        ('random_forest', RandomForestClassifier, {'n_estimators': 50, 'max_depth': 20, 'n_jobs': -1})
    ]
    
    print_progress(f"Models parallel aayi train cheyyunnu {n_jobs} CPU cores upayogichu...")
    
    # Models parallel aayi train cheyyuka
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate_model)(name, model_class, X_train, X_test, y_train, y_test, params)
        for name, model_class, params in tqdm(model_configs, desc="Models train cheyyunnu")
    )
    
    # Results dictionaryil convert cheyyuka
    results = {result['name']: {k: v for k, v in result.items() if k != 'name'} for result in results_list}
    
    # Overall results save cheyyuka
    with open(f"{MODEL_DIR}/training_results.json", 'w') as f:
        json.dump(results, f)
    
    return results

def main():
    """Optimizations upayogichu main training function"""
    print_progress("Optimized model training process start cheyyunnu...")
    
    # Faster traininginu vendi sample upayogikkano enn userine chodikkuka
    use_sample = input("Faster traininginu vendi smaller sample upayogikkano? (y/n): ").lower() == 'y'
    sample_size = 5000 if use_sample else None
    
    # Optional sampling upayogichu dataset load cheyyuka
    df = load_dataset(sample_size=sample_size)
    if df is None:
        return
    
    # Featuresum labelsum extract cheyyuka
    print_progress("Features extract cheyyunnu...")
    X = df['text'].values
    y = df['label'].values
    
    # Optimized TF-IDF vectorizer create cheyyuka (fewer features)
    print_progress("TF-IDF features create cheyyunnu (optimized)...")
    max_features = 3000 if use_sample else 5000  # Faster processinginu vendi features reduce cheyyuka
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,  # 2 documentsil thazhe appear cheyyunna terms ignore cheyyuka
        max_df=0.9  # 90% documentsil thazhe appear cheyyunna terms ignore cheyyuka
    )
    
    # Vectorizationinu vendi progress bar kanikkuka
    with tqdm(total=100, desc="Text vectorize cheyyunnu") as pbar:
        pbar.update(10)  # Start cheyyunnu
        X_tfidf = vectorizer.fit_transform(X)
        pbar.update(90)  # Complete cheyyunnu
    
    # Vectorizer save cheyyuka
    print_progress("Vectorizer save cheyyunnu...")
    with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Progress bar upayogichu data split cheyyuka
    print_progress("Dataset train/test setsil split cheyyunnu...")
    with tqdm(total=100, desc="Data split cheyyunnu") as pbar:
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        pbar.update(100)
    
    # Ella modelsum parallel aayi train cheyyuka
    print_progress("Optimized model training start cheyyunnu...")
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Overall accuracy calculate cheyyuka
    accuracies = [result['accuracy'] for result in results.values()]
    avg_accuracy = sum(accuracies) / len(accuracies)
    
    # Model metadata save cheyyuka
    print_progress("Model metadata save cheyyunnu...")
    metadata = {
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': len(df),
        'genuine_reviews': int(len(df[df['label'] == 0])),
        'fake_reviews': int(len(df[df['label'] == 1])),
        'feature_count': X_tfidf.shape[1],
        'average_accuracy': float(avg_accuracy),
        'best_model': max(results.items(), key=lambda x: x[1]['accuracy'])[0],
        'best_accuracy': float(max(result['accuracy'] for result in results.values())),
        'is_sample': use_sample
    }
    
    with open(f"{MODEL_DIR}/model_metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    print_progress(f"Training complete! Average accuracy: {avg_accuracy:.4f}")
    print(f"Best model: {metadata['best_model']} with accuracy: {metadata['best_accuracy']:.4f}")
    print(f"Models {os.path.abspath(MODEL_DIR)}il save cheythu")
    
    if use_sample:
        print("\nNOTE: Ningal traininginu vendi smaller sample upayogichu. Production useinu vendi,")
        print("full datasetil training cheyyuka, ningalkku kooduthal samayam undenkil.")

if __name__ == "__main__":
    main()