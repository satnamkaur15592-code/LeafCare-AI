"""
LeafCare-AI - Universal Leaf Disease Detection System
Supports 100+ Crops using Hybrid AI Vision + ML Approach
Author: LeafCare-AI Team
"""

import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LeafCare-AI | Universal Leaf Doctor",
    page_icon="🌿",
    layout="centered"
)

# --- HEADER ---
st.title("🌿 LeafCare-AI")
st.subheader("Universal Leaf Disease Detection System")
st.write("Supports 100+ Crops | Powered by AI + Machine Learning")
st.divider()

# --- API CONFIGURATION ---
# Get your free API key from: https://aistudio.google.com/app/apikey
# Store it in .env file as: GEMINI_API_KEY=your_key_here
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.warning("⚠️ API Key not found. Please create a .env file and add your GEMINI_API_KEY.")
else:
    genai.configure(api_key=API_KEY)

# --- LANGUAGE SELECTION ---
language = st.selectbox(
    "Select Language:",
    ["English", "Hindi"]
)

# --- IMAGE UPLOAD SECTION ---
st.subheader("📸 Upload Leaf Image")
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a single leaf for best accuracy"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    if st.button("🔍 Analyze Leaf", type="primary"):
        if not API_KEY:
            st.error("Please configure your API Key first.")
        else:
            with st.spinner("Analyzing leaf image... Please wait."):
                try:
                    model = genai.GenerativeModel("gemini-3.8-flash")

                    prompt = f"""
                    You are an expert Plant Pathologist with 20 years of experience.

                    Analyze the provided leaf image and respond in {language} language.
                    Provide a structured, professional report in the following format:

                    1. Plant Identification:
                       - Crop / Plant Name
                       - Scientific Name (if known)

                    2. Health Status:
                       - Healthy / Diseased

                    3. Disease Diagnosis:
                       - Disease Name
                       - Confidence Score (in %)

                    4. Cause:
                       - Primary reason for this disease

                    5. Treatment Plan:
                       - Recommended Chemical Treatment (with product name)
                       - Recommended Organic / Home Remedy

                    6. Prevention Measures:
                       - How to prevent this disease in future

                    7. Expert Advice:
                       - When to consult an agricultural expert

                    Important Instructions:
                    - If the image is not a leaf, clearly state: 'This is not a leaf image.'
                    - If the image is unclear, request a clearer image.
                    - Use simple, farmer-friendly language.
                    """

                    response = model.generate_content([prompt, image])

                    st.success("✅ Analysis Complete")
                    st.subheader("📋 Diagnosis Report")
                    st.write(response.text)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- FOOTER ---
st.divider()
st.caption("LeafCare-AI | Built with AI + ML | Supports 100+ Crops | Version 1.0")