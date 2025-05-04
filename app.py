import streamlit as st
from rag_pipeline import answer_query

st.set_page_config(page_title="AskLegalBot", layout="centered")
st.title("🧑‍⚖️ AskLegalBot - Your Legal Rights Assistant")

st.markdown("Ask a legal question based on the uploaded dataset.")
question = st.text_input("Enter your question here:")

if question:
    with st.spinner("Finding the best answer..."):
        answer = answer_query(question)
        st.success(answer)