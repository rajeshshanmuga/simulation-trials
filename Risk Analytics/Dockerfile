FROM python:3.7

WORKDIR /app

COPY Pip* /app/

RUN pip install --upgrade pip && \
    pip install pipenv && \
    pipenv install --deploy --system

COPY . /app/

EXPOSE 8501


# Use the following command to test it on Flask Server for dev purposes
CMD ["streamlit","run" "routes.py"]
