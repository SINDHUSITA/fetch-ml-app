FROM python:3.9
COPY . .
EXPOSE 8501
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "streamlit_main.py"]
