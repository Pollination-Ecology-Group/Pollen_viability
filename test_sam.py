from ultralytics import SAM
import sys

print("Loading SAM...")
model = SAM('sam_b.pt')
print("Model loaded successfully.")
print("Methods:", dir(model))
