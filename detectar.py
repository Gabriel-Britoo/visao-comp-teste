import torch
import cv2
import serial
import time

# comunicação serial
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # tempo para a conexão estabilizar

# carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)  # usa a webcam do notebook

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # realizar a detecção
    results = model(frame)
    detections = results.pandas().xyxy[0]  # converter para DataFrame

    # contar quantas pessoas foram detectadas
    num_pessoas = (detections['name'] == 'person').sum()

    # exibir na tela
    cv2.putText(frame, f'Pessoas: {num_pessoas}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('YOLOv5 Detecção de Pessoas', frame)

    # enviar sinal para o Arduino se houver pelo menos 1 pessoa detectada
    if num_pessoas > 2:
        arduino.write(b'1\n')  # enviar '1' para o Arduino
    else:
        arduino.write(b'0\n')  # enviar '0' para o Arduino

    time.sleep(1)    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()