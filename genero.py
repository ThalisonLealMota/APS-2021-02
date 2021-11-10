#importando bibliotecas
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# carregando modelos
model_humor = load_model('emotion_detection.h5')
model_genero = load_model('gender_detection.model')

# abrindo webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# labels de classificação
classes = ['Homem','Mulher']
emotion_dict = {0: "Raiva", 1: "Nojo", 2: "Medo", 3: "Feliz", 4: "Neutro", 5: "Triste", 6: "Surpreso"}

# loop pelos frames
while webcam.isOpened():

    # lê o frame da webcam 
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # aplica face detection
    face, confidence = cv.detect_face(frame)

    print(face)
    print(confidence)

    # loop pelos rostos detectados
    for idx, f in enumerate(face):

        # pega os pontos do retangulo da face detectada       
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # desenha o retangulo sobre a face detectada
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # corta a parte detectada
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # pre processamento para a detecção de genero e humor
        roi_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        

        # aplica detecçoes a face
        conf = model_genero.predict(face_crop)[0]
        prediction = model_humor.predict(cropped_img)[0]

        # pega a label com maior valor
        g_idx = np.argmax(conf)
        h_idx = np.argmax(prediction)
        g_label = classes[g_idx]
        h_label = emotion_dict[h_idx]
        g_label = "Genero: {} ({:.2f})".format(g_label, conf[g_idx] )
        h_label = "Emocao: {} ".format(h_label)

        # escreve a label abaixo do retangulo
        cv2.putText(frame, g_label, (startX, endY + 20),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(frame, h_label, (startX, endY+40), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, (255, 255, 255), 2)

    # mostra o frame
    cv2.imshow("gender detection", frame)

    # pressione "Q" para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# libera a webcam e destroi a janela
webcam.release()
cv2.destroyAllWindows()