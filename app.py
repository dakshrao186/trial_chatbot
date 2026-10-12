import streamlit as st
from chatbot import ask
#source .venv/bin/activate
# strealit run app.py
st.title("FootBall CHATBOT")

st.write("Ask any question related to football and its history.")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip():
        answer = ask(question)
        st.write("### Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")