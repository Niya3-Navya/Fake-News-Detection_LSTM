

# import streamlit as st
# import tensorflow as tf
# import pickle
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import re

# # Training wale same rules yahan apply honge
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'^.*?\(reuters\) - ', '', text) # Reuters check
#     text = re.sub(r'[^a-z ]', '', text)
#     return text

# # Load model and tokenizer
# @st.cache_resource # Baar-baar load hone se rokne ke liye
# def load_assets():
#     model = tf.keras.models.load_model("my_model.h5")
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     return model, tokenizer

# model, tokenizer = load_assets()

# st.set_page_config(page_title="Fake News Detector", page_icon="📰")
# st.title("📰 Fake News Detector")
# st.markdown("Enter news paragraph below to check if it's Real or Fake.")

# text_input = st.text_area("Paste News Text Here", height=200)

# if st.button("Analyze News"):
#     if text_input:
#         cleaned = clean_text(text_input)
#         seq = tokenizer.texts_to_sequences([cleaned])
#         padded = pad_sequences(seq, maxlen=300, padding='post', truncating='post')

#         prediction = model.predict(padded)[0][0]
        
#         # Display Results
#         if prediction > 0.5:
#             st.success(f"✅ **REAL NEWS** (Confidence: {prediction*100:.2f}%)")
#         else:
#             st.error(f"❌ **FAKE NEWS** (Confidence: {(1-prediction)*100:.2f}%)")
#     else:
#         st.warning("Please enter some text first!")



# import streamlit as st
# import tensorflow as tf
# import pickle
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import re
# import matplotlib.pyplot as plt

# # 1. Text Cleaning Function
# def clean_text(text):
#     text = text.lower()
#     # Reuters header aur special characters hatane ke liye
#     text = re.sub(r'^.*?\(reuters\) - ', '', text) 
#     text = re.sub(r'[^a-z ]', '', text)
#     return text

# # 2. Model aur Tokenizer Load karna
# @st.cache_resource 
# def load_assets():
#     model = tf.keras.models.load_model("my_model.h5")
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     return model, tokenizer

# model, tokenizer = load_assets()

# # 3. Streamlit UI Setup
# st.set_page_config(page_title="Fake News Detector", page_icon="📰")
# st.title("📰 Fake News Detector")
# st.markdown("Enter news paragraph below to check if it's Real or Fake.")

# text_input = st.text_area("Paste News Text Here", height=200)

# if st.button("Analyze News"):
#     if text_input:
#         # Preprocessing
#         cleaned = clean_text(text_input)
#         seq = tokenizer.texts_to_sequences([cleaned])
#         padded = pad_sequences(seq, maxlen=300, padding='post', truncating='post')

#         # Prediction
#         prediction = model.predict(padded)[0][0]
        
#         # Percentages nikalna
#         real_prob = float(prediction)
#         fake_prob = 1.0 - real_prob
        
#         # Result display karna
#         if prediction > 0.5:
#             st.success(f"✅ **REAL NEWS** (Confidence: {real_prob*100:.2f}%)")
#         else:
#             st.error(f"❌ **FAKE NEWS** (Confidence: {fake_prob*100:.2f}%)")

#         # --- PIE CHART SECTION ---
#         st.divider()
#         st.subheader("📊 Visual Analysis")
        
#         # Data taiyar karna
#         labels = ['Real', 'Fake']
#         sizes = [real_prob, fake_prob]
#         colors = ['#2ecc71', '#e74c3c'] # Green for Real, Red for Fake
#         explode = (0.1, 0) # Real wale hisse ko thoda bahar dikhane ke liye

#         # Chart banana
#         fig, ax = plt.subplots()
#         ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#                shadow=True, startangle=140, colors=colors)
#         ax.axis('equal') 

#         # Chart ko app mein dikhana
#         st.pyplot(fig)
        
#     else:
#         st.warning("Please enter some text first!")








import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import matplotlib.pyplot as plt

# 1. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'^.*?\(reuters\) - ', '', text) 
    text = re.sub(r'[^a-z ]', '', text)
    return text

# 2. Assets Load karna (Model, Tokenizer aur History)
@st.cache_resource 
def load_assets():
    model = tf.keras.models.load_model("my_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    try:
        with open("train_history.pkl", "rb") as f:
            history = pickle.load(f)
    except FileNotFoundError:
        history = None
    return model, tokenizer, history

model, tokenizer, history = load_assets()

# 3. Streamlit UI Setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")

# --- SIDEBAR (Overall Accuracy Graph yahan dikhayenge) ---
st.sidebar.title("📊 Model Performance")
if history:
    if st.sidebar.checkbox("Show Overall Accuracy Graph"):
        st.sidebar.markdown("prediction accuracy")
        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(history['accuracy'], label='Train Acc', color='#2ecc71')
        ax_acc.plot(history['val_accuracy'], label='Val Acc', color='#3498db')
        ax_acc.set_title('Accuracy over Epochs')
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        st.sidebar.pyplot(fig_acc)
else:
    st.sidebar.warning("Training history not found. Please run train_model.py first.")

# --- MAIN PAGE ---
st.markdown("Enter news paragraph below to check if it's Real or Fake.")
text_input = st.text_area("Paste News Text Here", height=200)

if st.button("Analyze News"):
    if text_input:
        cleaned = clean_text(text_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=300, padding='post', truncating='post')

        prediction = model.predict(padded)[0][0]
        real_prob = float(prediction)
        fake_prob = 1.0 - real_prob
        
        if prediction > 0.5:
            st.success(f"✅ **REAL NEWS** (Confidence: {real_prob*100:.2f}%)")
        else:
            st.error(f"❌ **FAKE NEWS** (Confidence: {fake_prob*100:.2f}%)")

        # Pie Chart Section
        st.divider()
        st.subheader("📊 Individual News Probability")
        
        labels = ['Real', 'Fake']
        sizes = [real_prob, fake_prob]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.1, 0)

        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
               shadow=True, startangle=140, colors=colors)
        ax.axis('equal') 
        st.pyplot(fig)
    else:
        st.warning("Please enter some text first!")