from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Flask app create cheyyuka, CORS enable cheyyuka
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Pre-trained modelsinu vendi path
MODEL_DIR = 'models'

# Ella pre-trained modelsum upayogichu oru review analyze cheyyuka
@app.route('/analyze', methods=['POST'])
def analyze_review():
    try:
        review_text = request.json.get('review_text', '')
        
        if not review_text:
            return jsonify({'error': 'No review text provided'}), 400
        
        # Vectorizer load cheyyuka
        with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Review text transform cheyyuka
        X = vectorizer.transform([review_text])
        
        # Ella modelsum load cheyyuka, predictions kittuka
        models = ['naive_bayes', 'linear_svc', 'svm', 'random_forest']
        predictions = {}
        
        for model_name in models:
            model_path = f"{MODEL_DIR}/{model_name}_model.pkl"
            if not os.path.exists(model_path):
                continue
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Prediction kittuka (1 fake, 0 genuine)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                prediction = model.predict(X)[0]
                confidence = proba[prediction]
            else:
                # Predict_proba illatha modelsinu vendi (LinearSVC pole)
                prediction = model.predict(X)[0]
                # Decision_function confidenceinu vendi upayogikkuka
                decision = abs(model.decision_function(X)[0])
                confidence = 0.5 + (0.5 * min(decision / 2, 1.0))  # 0.5-1.0 rangeil scale cheyyuka
            
            predictions[model_name] = {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'label': 'fake' if prediction == 1 else 'genuine'
            }
        
        # Ensemble prediction (confidence based cheytha weighted voting)
        weighted_votes = sum([pred['prediction'] * pred['confidence'] for pred in predictions.values()])
        total_weight = sum([pred['confidence'] for pred in predictions.values()])
        ensemble_prediction = 1 if weighted_votes / total_weight > 0.5 else 0
        
        # Average confidence calculate cheyyuka
        if ensemble_prediction == 1:
            # Fake predictioninu vendi, fake predict cheytha modelsinte average confidence
            fake_confidences = [pred['confidence'] for name, pred in predictions.items() if pred['prediction'] == 1]
            avg_confidence = sum(fake_confidences) / len(fake_confidences) if fake_confidences else 0.5
        else:
            # Genuine predictioninu vendi, genuine predict cheytha modelsinte average confidence
            genuine_confidences = [pred['confidence'] for name, pred in predictions.items() if pred['prediction'] == 0]
            avg_confidence = sum(genuine_confidences) / len(genuine_confidences) if genuine_confidences else 0.5
        
        # Decision influence cheytha key features extract cheyyuka
        feature_importance = extract_key_features(vectorizer, review_text, ensemble_prediction)
        
        return jsonify({
            'individual_predictions': predictions,
            'ensemble_prediction': {
                'prediction': ensemble_prediction,
                'confidence': avg_confidence,
                'label': 'fake' if ensemble_prediction == 1 else 'genuine'
            },
            'key_features': feature_importance
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_key_features(vectorizer, text, prediction):
    """Prediction influence cheytha key features extract cheyyuka"""
    # Vectorizeril ninnum feature names kittuka
    feature_names = vectorizer.get_feature_names_out()
    
    # Text transform cheythu feature indices kittuka
    X = vectorizer.transform([text])
    
    # Non-zero feature indicesum avayude valuesum kittuka
    non_zero_indices = X.nonzero()[1]
    feature_values = X.data
    
    # (feature, value) tuples list create cheyyuka
    features = [(feature_names[idx], value) for idx, value in zip(non_zero_indices, feature_values)]
    
    # Value (TF-IDF score) prakarama sort cheyyuka
    features.sort(key=lambda x: x[1], reverse=True)
    
    # Top features return cheyyuka
    top_features = features[:10]
    
    # Common fake review indicators
    fake_indicators = [
        "amazing", "best", "perfect", "awesome", "excellent", "outstanding",
        "terrible", "horrible", "awful", "worst",
        "!!!", "love it", "hate it", "never again", "must buy", "waste of money"
    ]
    
    # Common genuine review indicators
    genuine_indicators = [
        "however", "though", "although", "but", "pros", "cons",
        "specifically", "particular", "detail", "slightly", "somewhat", 
        "fairly", "quite", "rather"
    ]
    
    # Textil indicators check cheyyuka
    found_fake = [word for word in fake_indicators if word.lower() in text.lower()]
    found_genuine = [word for word in genuine_indicators if word.lower() in text.lower()]
    
    return {
        'top_features': top_features,
        'fake_indicators': found_fake,
        'genuine_indicators': found_genuine
    }

# Dataset statistics kittan vendi route
@app.route('/dataset_stats', methods=['GET'])
def get_dataset_stats():
    """Pre-trained modelsinte kurichulla statistics kittuka"""
    metadata_file = os.path.join(MODEL_DIR, 'model_metadata.json')
    
    if not os.path.exists(metadata_file):
        # Metadata file illenkil, oru basic one create cheyyuka
        metadata = {
            'models': [],
            'average_accuracy': 0.85,  # Default value
            'dataset_size': 'Unknown',
            'genuine_reviews': 'Unknown',
            'fake_reviews': 'Unknown'
        }
        
        # Models exist cheyyunnath check cheyyuka
        for model_name in ['naive_bayes', 'linear_svc', 'svm', 'random_forest']:
            if os.path.exists(os.path.join(MODEL_DIR, f"{model_name}_model.pkl")):
                metadata['models'].append(model_name)
        
        # Metadata save cheyyuka
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': f'Error reading metadata: {str(e)}'}), 500

# Connection testinginu vendi ping endpoint add cheyyuka
@app.route('/ping', methods=['GET'])
def ping():
    """Server nadakkunnundenn test cheyyan vendi simple endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)