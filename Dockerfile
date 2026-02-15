# Use Python 3.9 as base
FROM python:3.9

# 1. Install FFmpeg (Required for librosa/pydub to read audio files)
RUN apt-get update && apt-get install -y ffmpeg

# 2. Set up working directory
WORKDIR /code

# 3. Create a cache directory for AI models with write permissions
# (Hugging Face spaces often crash if they can't write to cache)
RUN mkdir -p /code/cache
ENV XDG_CACHE_HOME=/code/cache
ENV HF_HOME=/code/cache
RUN chmod 777 /code/cache

# 4. Install Python Dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy your application code
COPY . /code

# 6. Start the server
# Using port 7860 is standard for Hugging Face Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]