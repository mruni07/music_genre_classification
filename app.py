import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load models
model = pickle.load(open('music_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Load dataset to get mean values
df = pd.read_csv('music_genre_spectral_features_large.csv')
df = df.drop(columns=['filename', 'label'])

# Calculate mean for each feature
feature_means = df.mean().to_dict()
feature_names = df.columns.tolist()

st.set_page_config(page_title="Music Genre Predictor", page_icon="🎵")
st.title("🎵 Music Genre Predictor")
st.caption(f"Enter 20 key features. Remaining {len(feature_names) - 20} features will use dataset averages for accuracy")

st.subheader("📝 Enter 20 Key Audio Features")

col1, col2, col3, col4 = st.columns(4)

user_inputs = {}

# Feature Group 1: Basic (5 features)
with col1:
    st.markdown("**Basic Features**")
    user_inputs['length'] = st.number_input("Length", value=200.0)
    user_inputs['tempo'] = st.number_input("Tempo", value=120.0)
    user_inputs['rms_mean'] = st.number_input("RMS Mean", value=0.08)
    user_inputs['rms_var'] = st.number_input("RMS Variance", value=0.03)
    user_inputs['zero_crossing_rate_mean'] = st.number_input("Zero Crossing Rate", value=0.07)

# Feature Group 2: Spectral (5 features)
with col2:
    st.markdown("**Spectral Features**")
    user_inputs['spectral_centroid_mean'] = st.number_input("Spectral Centroid Mean", value=2500.0)
    user_inputs['spectral_centroid_var'] = st.number_input("Spectral Centroid Var", value=200.0)
    user_inputs['spectral_bandwidth_mean'] = st.number_input("Spectral Bandwidth Mean", value=2000.0)
    user_inputs['spectral_bandwidth_var'] = st.number_input("Spectral Bandwidth Var", value=150.0)
    user_inputs['rolloff_mean'] = st.number_input("Rolloff Mean", value=5000.0)

# Feature Group 3: Chroma & Harmony (5 features)
with col3:
    st.markdown("**Chroma & Harmony**")
    user_inputs['chroma_stft_mean'] = st.number_input("Chroma STFT Mean", value=0.5)
    user_inputs['chroma_stft_var'] = st.number_input("Chroma STFT Var", value=0.12)
    user_inputs['harmony_mean'] = st.number_input("Harmony Mean", value=0.5)
    user_inputs['harmony_var'] = st.number_input("Harmony Var", value=0.1)
    user_inputs['perceptr_mean'] = st.number_input("Perceptr Mean", value=0.3)

# Feature Group 4: MFCC (5 features)
with col4:
    st.markdown("**MFCC Features**")
    user_inputs['mfcc1_mean'] = st.number_input("MFCC1 Mean", value=0.0)
    user_inputs['mfcc2_mean'] = st.number_input("MFCC2 Mean", value=0.0)
    user_inputs['mfcc3_mean'] = st.number_input("MFCC3 Mean", value=0.0)
    user_inputs['mfcc4_mean'] = st.number_input("MFCC4 Mean", value=0.0)
    user_inputs['mfcc5_mean'] = st.number_input("MFCC5 Mean", value=0.0)

# Get all features in correct order
inputs = []
filled_from_dataset = 0
filled_from_user = 0

for feature in feature_names:
    if feature in user_inputs:
        # Use user input value
        inputs.append(user_inputs[feature])
        filled_from_user += 1
    else:
        # Use dataset mean for remaining features
        inputs.append(feature_means[feature])
        filled_from_dataset += 1

st.info(f"📊 Using {filled_from_user} features from your input + {filled_from_dataset} features from dataset averages")

# PREDICTION
if st.button("🎯 Predict Genre", type="primary"):
    data = np.array(inputs).reshape(1, -1)
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    result = le.inverse_transform([pred])[0]
    
    st.success(f"### 🎵 Predicted Genre: **{result}**")
    

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(data_scaled)[0]
        confidence = probs[pred] * 100
        st.progress(int(confidence))
        st.caption(f"Confidence: {confidence:.1f}%")