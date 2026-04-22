
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

 ##  1. Real-Time News Prediction:-
   This screen represents the core functionality of the application, where the user interacts
   with the trained LSTM model to verify news authenticity. 
 
 Features and Functionalities:
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


## 2. Visual Probability Analysis (Pie Chart):-

This screen demonstrates the visual analytics component of the application, designed to
provide users with a deeper understanding of the model's decision-making process through
graphical representation.


Intuitive Color Scheme: 
* The application uses a standardized color palette for clarity:
Green (#2ecc71): Representing "Real" news, symbolizing authenticity.
Red (#e74c3c): Representing "Fake" news, symbolizing a warning or caution.



<img width="765" height="424" alt="image" src="https://github.com/user-attachments/assets/097fbf04-7a91-45f5-b847-371cf657f7a7" />

## 3. Fake News Classification:-
* This interface demonstrates the system’s ability to successfully identify and flag fabricated
  content. Upon processing the input text, the model generates a prominent Red Error Banner
  labeled "FAKE NEWS".

* This visual cue is designed to provide an immediate warning to the
  user, accompanied by a calculated confidence score (e.g., 100.00%), which reflects the high
  degree of mathematical certainty determined by the LSTM network's final layer.

 * To reinforce the textual classification, the application provides a corresponding Visual
   Analysis via a Pie Chart


<img width="758" height="422" alt="image" src="https://github.com/user-attachments/assets/a1a73dd0-b73e-4205-88d0-9b6ce04cae37" />

<img width="765" height="421" alt="image" src="https://github.com/user-attachments/assets/bad445f8-844c-4273-b2bb-48c87e715f06" />


## 4. Sidebar Navigation & Performance Control:-
 * This section of the application demonstrates the secondary interface layer, designed to provide
   users with access to the model's technical background without cluttering the main prediction
   area.
   
 * Model Training Performance (Accuracy Graph):-
   This graphical representation provides a technical overview of how the deep learning model improved
   its predictive capabilities over time during the training phase.
   
 * Dual-Metric Comparison: The visualization tracks two critical metrics simultaneously to ensure the
   model is learning correctly:
    (A). Train Acc (Green Line): Represents the accuracy achieved on the training dataset. It shows how
         well the model is learning the patterns from the data it has already seen.
    (B). Val Acc (Blue Line): Represents "Validation Accuracy" on a separate, unseen dataset. This is the
          most important metric as it proves the model can generalize and predict news it hasn't
          encountered before.
   
 *  Performance Stability: The Y-axis represents the Accuracy Score (ranging from 0.0 to 1.0). The
    graph shows that both lines converge toward a high accuracy (above 95%), which indicates a highly
    successful training session.



<img width="765" height="422" alt="image" src="https://github.com/user-attachments/assets/91a0a45b-3755-463b-bce4-adaa316148a4" />


## 3. Results and Analysis:-

  * Confusion Matrix:-The Confusion Matrix serves as a comprehensive evaluation tool that summarizes the
     performance of the classification algorithm beyond simple accuracy.


<img width="720" height="540" alt="image" src="https://github.com/user-attachments/assets/8c11686b-dd31-4fc0-9c66-948fd684d21c" />





