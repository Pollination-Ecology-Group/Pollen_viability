
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install additional dependencies
RUN pip install boto3 pandas

# Copy application code
COPY src/ ./

# Ensure directories have correct permissions for UID 1000 (which Kubernetes uses)
RUN chown -R 1000:1000 /app && \
    mkdir -p /ultralytics/runs && \
    chown -R 1000:1000 /ultralytics

USER 1000

# Run the detection script
CMD ["python", "run_detection.py"]
