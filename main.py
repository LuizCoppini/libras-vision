import cv2
import mediapipe as mp

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Abre a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    # Volta para BGR para exibir no OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Desenha os pontos da mão
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Detecção de Mãos", image)

    if cv2.waitKey(1) & 0xFF == 27:  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
