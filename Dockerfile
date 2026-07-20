FROM python:3.14-slim

WORKDIR /workspace

COPY requirements.lock .
RUN pip install --no-cache-dir --requirement requirements.lock

COPY . .

EXPOSE 8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app/project_streamlit_dashboard.py"]
