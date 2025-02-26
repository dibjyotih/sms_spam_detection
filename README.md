# Spam Detection using Machine Learning

This project is an SMS/Email spam classifier built using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. The model is trained using **TF-IDF vectorization** and a **Naïve Bayes classifier** to classify messages as **Spam** or **Not Spam**.

## Features
- Preprocessing of text data (lowercasing, tokenization, stopword removal, stemming)
- TF-IDF vectorization for feature extraction
- Naïve Bayes classification for spam detection
- Web app using **Streamlit** for easy user interaction

## Dataset
The dataset used for training is stored in `spam.csv`, containing labeled SMS messages:
- `ham`: Not Spam
- `spam`: Spam

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

## Model Training
If you need to retrain the model, run the Jupyter notebook `sms_spam_detect.ipynb`:
```bash
jupyter notebook sms_spam_detect.ipynb
```
This will train the model, vectorize text using TF-IDF, and save:
- `vectorizer.pkl` (TF-IDF vectorizer)
- `model.pkl` (Trained Naïve Bayes model)

## Running the Web App
To run the Streamlit web app:
```bash
streamlit run app.py
```
Enter an SMS or email message, and the app will predict if it's **Spam** or **Not Spam**.

## File Structure
```
spam-detection/
│── app.py                 # Streamlit web app
│── sms_spam_detect.ipynb  # Jupyter Notebook for model training
│── spam.csv               # Dataset
│── vectorizer.pkl         # TF-IDF vectorizer
│── model.pkl              # Trained Naïve Bayes model
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

## Dependencies
- Python 3.7+
- Scikit-learn
- NLTK
- Pandas
- Streamlit

## Future Improvements
- Improve accuracy with deep learning models (LSTM, BERT)
- Deploy using Flask/FastAPI
- Enhance preprocessing techniques

## License
This project is licensed under the MIT License.

## Resources
Youtube
kaggle
Github

