
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install additional dependencies
RUN pip install boto3 pandas

# Copy application code
COPY run_detection.py train_model.py ./

# Run the detection script
CMD ["python", "run_detection.py"]
