import streamlit as st
import os
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import pandas as pd
import zipfile
from dotenv import load_dotenv
from io import BytesIO
import os

## page config

st.set_page_config(
    page_title="Resume → CSV",
    page_icon="📄",
    layout="centered"
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment")

## structure--
class ResumeData(TypedDict):
    summary: Annotated[str, "Provide a concise professional summary from the resume"]
    education: Annotated[str, "List education details (degrees, institutions, years)"]
    projects: Annotated[str, "Summarize key projects mentioned in the resume"]
    skills: Annotated[str, "Extract technical and soft skills from the resume"]
    experience: Annotated[str, "Extract total years of professional experience"]
    contact_details: Annotated[str, "Provide contact details (email, phone, LinkedIn if available)"]

## model---
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0)
fm = model.with_structured_output(ResumeData)

## ui
st.title("📝 :red[Resume] :blue[TO] :green[CSV] 𝄜")

uploaded_zip = st.file_uploader("Upload a ZIP file containing resumes (PDFs)", type=["zip"])

results = {}

## process

from io import BytesIO

...

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]

        for pdf_file in pdf_files:
            with z.open(pdf_file) as f:
                pdf_bytes = f.read()
                pdf_stream = BytesIO(pdf_bytes)

                pdf_reader = PdfReader(pdf_stream)
                text = "".join(
                    page.extract_text() or "" 
                    for page in pdf_reader.pages
                )

                if not text.strip():
                    continue

                response = fm.invoke(text)
                results[pdf_file] = response
## display
    st.subheader("📑 Extracted Resume Data (Dictionary)")
    st.json(results)

     
    df = pd.DataFrame.from_dict(results, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "filename"}, inplace=True)

    st.subheader("📊 Resume Data (DataFrame)")
    st.dataframe(df)

     
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="resume_data.csv",
        mime="text/csv",
    )