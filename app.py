import streamlit as st
import pickle
import nltk
import warnings
warnings.filterwarnings('ignore')
import re
from string import punctuation
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

def txt_process(input_text):
  input_text = ''.join([x for x in input_text if x not in punctuation])
  input_text = re.sub(r"[^A-Za-z0-9]", " ", input_text)
  input_text = re.sub(r"\'s", " ", input_text)

  input_text = input_text.split()
  input_text = [x for x in input_text if not x in stop_words]
  input_text = " ".join(input_text)

  return(input_text)
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def find_similarity_score(q1,q2):
  Q1_C= txt_process(q1)
  Q2_C = txt_process(q2)
  Q1_C = tokenizer.texts_to_sequences([Q1_C])
  Q2_C = tokenizer.texts_to_sequences([Q2_C])
  Q_final = Q1_C[0] + Q2_C[0]
  Q_Test = sequence.pad_sequences([Q_final], maxlen = 500)
  Prob=model.predict(Q_Test)
  print(Prob)
  return Prob[0]>0.4


st.title("Duplicate Question Detection")

text1 = st.text_area("Enter the first question:")
text2 = st.text_area("Enter the second question:")

if st.button("Check Duplicate"):
    if text1 and text2:
        result = find_similarity_score(text1, text2)
        if result:
            st.write("The questions are duplicates.")
        else:
            st.write("The questions are not duplicates.")
    else:
        st.write("Please enter both questions.")
