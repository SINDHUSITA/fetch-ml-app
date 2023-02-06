import streamlit as st
from PIL import Image
from io import BytesIO
import pandas as pd
from io import StringIO
from model.model_inference import ModelInference



st.set_page_config(layout="wide", page_title="Forecast Application")

st.write("## Forecast data trends")
st.write(
    "Try uploading some sample data at the sidebar."
)
st.sidebar.write("## Upload and download :gear:")


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def temp_image(upload):
    ModelInference(input_data_path=upload,).model_result()
    image = Image.open("data/input_data.png")
    col1.write("Data Before Forecast:")
    col1.image(image)


    col2.write("Data After Forecast")
    result_image = Image.open("results/output_result.png")
    col2.image(result_image)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download forecast image", convert_image(image), "forecast.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an input excel file", type=["csv"])

if my_upload is not None:
    temp_image(upload=my_upload)
else:
    temp_image(upload="data_daily.csv")