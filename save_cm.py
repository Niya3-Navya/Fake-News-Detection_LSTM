import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data load karein
with open('confusion_matrix.pkl', 'rb') as f:
    cm = pickle.load(f)

# 2. Sundar Heatmap banayein
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'])

plt.title('Confusion Matrix - Fake News Detector')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

# 3. Photo save karein
plt.savefig('confusion_matrix_report.png')
print("✅ Image saved as 'confusion_matrix_report.png'. Aap ise report mein use kar sakte hain!")