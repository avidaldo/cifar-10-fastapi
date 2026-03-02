# Use the official Python 3.10 image
FROM python:3.10-slim

# 1. Create the user immediately (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 2. Set the working directory to the user's home
WORKDIR $HOME/app

# 3. Copy requirements first to leverage Docker layer caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Create necessary directories INSIDE the correct user path
RUN mkdir -p models static/uploads

# 5. Copy the rest of the application
COPY --chown=user:user . $HOME/app

# Command to run the FastApi application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
