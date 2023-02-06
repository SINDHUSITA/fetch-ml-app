FROM python:3.9
COPY . /src/app
WORKDIR /src/app
EXPOSE 8501
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "streamlit_main.py", "--server.port=8501", "--server.address=0.0.0.0"]