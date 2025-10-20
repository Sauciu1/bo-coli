# Base image is Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

COPY src/ ./src/
COPY pyproject.toml ./




RUN pip install poetry
RUN poetry lock
RUN poetry install --no-root



# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8989
ENV PYTHONPATH=/app


CMD ["poetry", "run", "streamlit", "run", "./src/ui/main_ui.py", "--server.port=8989", "--server.address=0.0.0.0"]


EXPOSE 8989
