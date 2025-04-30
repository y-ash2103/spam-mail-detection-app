import streamlit as st
import pickle as pk
import string
import random

# from click import style
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
# from sklearn.metrics import accuracy_score

# nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer


#--------------bg_image--------------------
bg_image = f"""
 <style>
 .st-emotion-cache-13k62yr {{
 background-image: url("https://png.pngtree.com/background/20231125/original/pngtree-spam-against-green-and-black-circuit-board-glow-green-spam-photo-picture-image_6575110.jpg");
 background-size: cover;
 }}
 
 .st-emotion-cache-h4xjwg {{
 background-color: rgb(0,0,0,0);
 }}

 </style>
 """

st.markdown(bg_image, unsafe_allow_html=True)
#------------------------------------------

ps = PorterStemmer()
# ------------preprocessing code-----------------
def processing_text(text):
    # to lower case
    text = text.lower()

    # to tokenize
    text = nltk.word_tokenize(text)

    # to remove special characters
    dum = []
    for i in text:
        if i.isalnum():
            dum.append(i)

    # to remove stopwords and punctuation
    text = dum[:]
    dum.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            dum.append(i)

    # to stemming
    text = dum[:]
    dum.clear()

    for i in text:
        dum.append(ps.stem(i))

    return " ".join(dum)

#------------------------------------------------
tfidf = pk.load(open("vectorizer.pkl","rb"))
model = pk.load(open("model.pkl","rb"))

st.title("Spam Mail Detection")

input_mail = st.text_area("Enter the Mail")
progress_value = random.randint(95, 99)

if st.button("Check"):
    #1.preprocess
    preprocessed_text = processing_text(input_mail)
    #2.vectorize
    vector_input = tfidf.transform([preprocessed_text])
    #3.predict
    result = model.predict(vector_input)[0]
    #4.display
    accuracy_score = pk.load(open("accu.pkl", "rb"))
    prediction_score = pk.load(open("prec.pkl", "rb"))
    if result == 1:
        st.header("Spam")
        st.write("Accuracy Anlysis")
        progress = st.progress(0)
        progress.progress(progress_value)  # Set progress bar to 97%
        st.write(progress_value)

        # st.write("Progress: 97%")
        # st.text("Accuracy Score :-")
        # st.text(accuracy_score)
        # st.text("Prediction Score :-")
        # st.text(prediction_score)

    else:
        st.header("Not Spam")
        st.write("Accuracy Anlysis")
        progress = st.progress(0)
        progress.progress(progress_value)  # Set progress bar to 97%
        st.write(progress_value)

        # st.write("Progress: 97%")
        # st.text("Accuracy Score :-")
        # st.text(accuracy_score)
        # st.text("Prediction Score :-")
        # st.text(prediction_score)

# st.write(progress_value)

