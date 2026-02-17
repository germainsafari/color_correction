print("--- STARTING IMPORT TEST ---")

try:
    print("1. Attempting to import Numpy...")
    import numpy
    print("   -> OK: Numpy loaded.")
except Exception as e:
    print(f"   -> FATAL: {e}")

try:
    print("2. Attempting to import OpenCV...")
    import cv2
    print("   -> OK: OpenCV loaded.")
except Exception as e:
    print(f"   -> FATAL: {e}")

try:
    print("3. Attempting to import FastAPI...")
    import fastapi
    print("   -> OK: FastAPI loaded.")
except Exception as e:
    print(f"   -> FATAL: {e}")

try:
    print("4. Attempting to import Uvicorn...")
    import uvicorn
    print("   -> OK: Uvicorn loaded.")
except Exception as e:
    print(f"   -> FATAL: {e}")

try:
    print("5. Attempting to import MediaPipe...")
    import mediapipe
    print("   -> OK: MediaPipe loaded.")
except Exception as e:
    print(f"   -> FATAL: {e}")

print("--- END OF TEST ---")
