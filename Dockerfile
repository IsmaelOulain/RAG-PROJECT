FROM python:3.11

WORKDIR /UNI-CHATBOT

COPY ./requirements.txt /UNI-CHATBOT/requirements.txt

RUN pip install -r /UNI-CHATBOT/requirements.txt

RUN pip install --upgrade pip

COPY . /UNI-CHATBOT

EXPOSE 8000

CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]