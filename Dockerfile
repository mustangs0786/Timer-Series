FROM python:3.7.2-slim
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r Requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["Main.py"]