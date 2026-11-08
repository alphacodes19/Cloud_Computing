FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir boto3==1.34.131

COPY worker/label.py ./label.py
COPY prompts/teacher_prompt.txt ./prompts/teacher_prompt.txt
COPY data/schema.json ./data/schema.json

ENV PYTHONUNBUFFERED=1

CMD ["python", "label.py"]
