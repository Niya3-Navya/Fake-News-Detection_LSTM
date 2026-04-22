
# Fake News Detection using LSTM 🕵️‍♂️📰

## 📌 Project Overview
This project leverages **Natural Language Processing (NLP)** and **Long Short-Term Memory (LSTM)** neural networks to distinguish between real and fake news articles. With the rapid spread of misinformation, this tool provides a high-accuracy automated classification system.

## 🚀 Key Features
*   **Sequential Data Processing:** Utilizes LSTM to capture long-term dependencies and contextual meaning in news text.
*   **Preprocessing Pipeline:** Includes tokenization, stopword removal, and padding to prepare raw text for the model.
*   **High Performance:** Reached an accuracy of **99.8%** on the validation set during testing.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Deep Learning:** TensorFlow / Keras
*   **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, NLTK

## 📊 Dataset
The model was trained on a combined dataset of ~44,000 news articles, split into training and testing sets.

## ⚙️ How to Run
1. Clone the repo: `git clone https://github.com`
2. Run the notebook/script: `python detect_news.py`

***Note:*** Visualize the working using Fake_News_ML.ipynb ( download True.csv and Fake.csv before ) from Colab


## ScreenShots of System:-

 ##  1. Features and Functionalities:
  * Interactive Text Input: The user is provided with a large text area labeled "Paste News
    Text Here". This allows for the input of full-length news paragraphs or articles directly
    from any external source.

  
  * One-Click Analysis: The "Analyze News" button triggers the entire backend pipeline,
   including text cleaning, tokenization, and model inference, with a single click.


  * Real-Time Classification Output: As requested, the primary output is displayed in a
    single, clear line using a color-coded status banner.
    A Green Success Banner indicates that the model has classified the input as "REAL
    NEWS".
    The output also includes a verification emoji (✅) for quick visual confirmation.


  * Confidence Score Display: Along with the classification, the system displays a
    Confidence Percentage (e.g., 100.00%). This represents the mathematical probability
    calculated by the model's Sigmoid activation layer, informing the user how certain the
    AI is about its prediction.


  * Dynamic Sidebar: On the left, a sidebar is visible with a "Show Accuracy Graph" toggle,
    allowing the user to switch between the prediction view and the model's overall
    training performance metrics.
 


<img width="763" height="427" alt="image" src="https://github.com/user-attachments/assets/a9706251-d4d2-4a3f-885a-e3d3167ee82a" />



## Intuitive Color Scheme: 
The application uses a standardized color palette for clarity:
Green (#2ecc71): Representing "Real" news, symbolizing authenticity.
Red (#e74c3c): Representing "Fake" news, symbolizing a warning or caution.



<img width="765" height="424" alt="image" src="https://github.com/user-attachments/assets/097fbf04-7a91-45f5-b847-371cf657f7a7" />





<img width="758" height="422" alt="image" src="https://github.com/user-attachments/assets/a1a73dd0-b73e-4205-88d0-9b6ce04cae37" />

<img width="503" height="329" alt="image" src="https://github.com/user-attachments/assets/317b8c41-f1cb-42de-94fe-5bf2021fe078" />


##Model Training Performance (Accuracy Graph):-
This graphical representation provides a technical overview of how the deep learning model improved
its predictive capabilities over time during the training phase.

<img width="765" height="422" alt="image" src="https://github.com/user-attachments/assets/91a0a45b-3755-463b-bce4-adaa316148a4" />
