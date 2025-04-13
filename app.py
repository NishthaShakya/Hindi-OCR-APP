import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import stanza

# Indic NLP Setup
from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize

# 🔧 Set this to your actual path
INDIC_NLP_RESOURCES =  "C:/Users/Asus/indic_nlp_resources"  
common.set_resources_path(INDIC_NLP_RESOURCES)
loader.load()

# Download Stanza Hindi model (run only once)
stanza.download('hi')

# Initialize NLP pipeline
nlp = stanza.Pipeline(lang='hi', processors='tokenize,pos', use_gpu=False)

st.title("📝 Handwritten Hindi Text Recognition")
st.write("Upload a handwritten Hindi text image. This app extracts text using EasyOCR, then performs tokenization and POS tagging.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to array
    image_array = np.array(image)

    # Perform OCR
    with st.spinner("Extracting text..."):
        reader = easyocr.Reader(['hi'], gpu=False)
        result = reader.readtext(image_array)

    # Show extracted text
    st.subheader("📜 Extracted Text:")
    extracted_text = ""
    if result:
        for detection in result:
            st.write(detection[1])
            extracted_text += detection[1] + " "
    else:
        st.warning("No text detected.")

    # Tokenization
    st.subheader("🔍 Tokenization:")
    tokens = list(indic_tokenize.trivial_tokenize(extracted_text.strip(), lang='hi'))
    st.write(tokens)

    # POS Tagging
    st.subheader("🔤 POS Tagging:")
    doc = nlp(extracted_text.strip())
    for sentence in doc.sentences:
        for word in sentence.words:
            st.write(f"**{word.text}** → *{word.pos}*")
