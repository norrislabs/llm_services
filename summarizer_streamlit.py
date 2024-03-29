import argparse
import streamlit as st
from summarizer import summarize_article


ap = argparse.ArgumentParser()
ap.add_argument('model', default="", help="model file or API key")
ap.add_argument("gpu", type=int, default=0, help="number of gpu layers")
args = vars(ap.parse_args())

# Set page title
st.set_page_config(page_title="Article Summarizer", page_icon="📜", layout="wide")

# Set title
st.title("Article Summarizer", anchor=False)
st.header("Summarize Articles with AI", anchor=False)

# Input URL
st.divider()
url = st.text_input("Enter Article URL", value="")

# Download
st.divider()
if url:
    with st.status("Processing...", state="running", expanded=True) as status:
        st.write("Summarizing Article...")
        summary, time_taken = summarize_article(url, args['model'], args['gpu'])
        status.update(label=f"Finished - Time Taken: {round(time_taken, 1)} seconds", state="complete")

    # Show Summary
    st.subheader("Summary:", anchor=False)
    st.write(summary)
