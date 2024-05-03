FROM python:3.10.1-slim
WORKDIR /app
COPY testingml.py .

# Install system dependencies
RUN pip install opencv-python
ENV DISPLAY=:0
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python dependencies


# Copy your application code


# Set the entry point
CMD ["python", "testingml.py"]
