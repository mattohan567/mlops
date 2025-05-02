# Use official Python slim image
FROM python:3.11-slim

# Create app dir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app and model artifact
COPY redditApp.py reddit_model_pipeline.joblib ./

# Expose port
EXPOSE 8000

# Launch with Uvicorn
CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]