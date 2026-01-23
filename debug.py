print("--- INICIANDO TESTE DE IMPORTS ---")

try:
    print("1. Tentando importar Numpy...")
    import numpy
    print("   -> SUCESSO: Numpy carregado.")
except Exception as e:
    print(f"   -> ERRO FATAL no Numpy: {e}")

try:
    print("2. Tentando importar OpenCV...")
    import cv2
    print("   -> SUCESSO: OpenCV carregado.")
except Exception as e:
    print(f"   -> ERRO FATAL no OpenCV: {e}")

try:
    print("3. Tentando importar Streamlit...")
    import streamlit
    print("   -> SUCESSO: Streamlit carregado.")
except Exception as e:
    print(f"   -> ERRO FATAL no Streamlit: {e}")

try:
    print("4. Tentando importar MediaPipe...")
    import mediapipe
    print("   -> SUCESSO: MediaPipe carregado.")
except Exception as e:
    print(f"   -> ERRO FATAL no MediaPipe: {e}")

print("--- FIM DO TESTE ---")