import streamlit as st

st.set_page_config(
    page_title="Findora",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Findora")
st.caption("Search by memory, not metadata.")

st.markdown("---")

query = st.text_input(
    "Search your memories",
    placeholder="A magenta flower with a white background..."
)

if query:
    st.write("Your search:", query)