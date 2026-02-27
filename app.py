import streamlit as st
import joblib

# ✅ Cache model & vectorizer so they load once per server session
@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("vectorizer.pkl")

model = load_model()
vectorizer = load_vectorizer()

# Streamlit UI
st.title("📰 Fake News Detection System")
st.write("Enter news text to check whether it is Fake or Real.")

news = st.text_area("Enter News Article")

# ✅ Only run prediction when user clicks
if st.button("Predict"):

    if news.strip() != "":
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)

        if prediction[0] == "REAL":
            st.success("✅ Real News")
        else:
            st.error("🚨 Fake News")
    else:
        st.warning("Please enter text")