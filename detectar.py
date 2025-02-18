import torch
import cv2

# Carregar o modelo treinado
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', source='local')

# Iniciar a captura da câmera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção
    results = model(frame)

    # Filtrar apenas pessoas (classe 0 no COCO dataset)
    detections = results.pandas().xyxy[0]  # Converter para pandas DataFrame
    num_pessoas = (detections['name'] == 'person').sum()

    # Exibir resultado
    cv2.putText(frame, f'Pessoas: {num_pessoas}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('YOLOv5 Detecção de Pessoas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()