# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy everything to the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the FastAPI app using uvicorn
CMD ["fastapi", "run", "main.py"]
