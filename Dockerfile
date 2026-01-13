
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install additional dependencies
RUN pip install boto3 pandas

# Copy application code
COPY src/ ./

# Create a non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /ultralytics/runs && \
    chown -R appuser:appuser /ultralytics

USER 1000

# Run the detection script
CMD ["python", "run_detection.py"]
