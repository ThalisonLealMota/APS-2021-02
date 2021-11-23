#importando bibliotecas
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import os
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# carregando modelos
model_humor = load_model('emotion_detection.h5')
model_genero = load_model('gender_detection.model')

# labels de classificação
classes = ['Masculino','Feminino']
emotion_dict = {0: "Raiva", 1: "Desgosto", 2: "Medo", 3: "Feliz", 4: "Neutro", 5: "Triste", 6: "Surpreso"}

# loop pelos frames

def detectar(frame,classes, emotion_dict, model_humor, model_genero):
    face, confianca = cv.detect_face(frame)

    # loop pelos rostos detectados
    for idx, f in enumerate(face):

        # pega os pontos do retangulo da face detectada       
        (comecoX, comecoY) = f[0], f[1]
        (fimX, fimY) = f[2], f[3]

        # desenha o retangulo sobre a face detectada
        cv2.rectangle(frame, (comecoX,comecoY), (fimX,fimY), (0,255,0), 2)

        # corta a parte detectada
        face_crop = np.copy(frame[comecoY:fimY,comecoX:fimX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # pre processamento para a detecção de genero e humor
        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_crop, (48, 48)), -1), 0)
        
        # aplica detecçoes a face
        conf = model_genero.predict(face_crop)[0]
        prediction = model_humor.predict(cropped_img)[0]

        # pega a label com maior valor
        gender_idx = np.argmax(conf)
        humor_idx = np.argmax(prediction)
        gender_label = classes[gender_idx]
        humor_label = emotion_dict[humor_idx]
        gender_label = "Genero: {} ({:.2f})".format(gender_label, conf[gender_idx] )
        humor_label = "Emocao: {} ".format(humor_label)

        # escreve a label abaixo do retangulo
        cv2.putText(frame, gender_label, (comecoX, comecoY - 20),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        cv2.putText(frame, humor_label, (comecoX, comecoY - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    
    return frame


def detecta_video():
    # abrindo webcam
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Não consegue abrir a webcam")
        exit()

    while webcam.isOpened():

        # lê o frame da webcam 
        status, frame = webcam.read()

        if not status:
            print("Não consegue ler o frame")
            exit()

        # aplica face detection
        
        frame = detectar(frame, classes, emotion_dict, model_humor, model_genero)

        # mostra o frame
        cv2.imshow("Deteccao Webcam", frame)

        # pressione "Q" para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()
    cv2.destroyAllWindows()


def detecta_imagem():
    #Encontrando imagem
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Selecione uma Imagem", filetypes=(("Arquivos JPG", "*.jpg"), ("Arquivos PNG", "*.png"), ("Todos os arquivos", "*.*")))

    img = cv2.imread(fln)
    
    img = detectar(img, classes, emotion_dict, model_humor, model_genero)

    img = cv2.resize(img,(500,500))
    cv2.imshow("Deteccao Imagem", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Tkinter Menu
root = Tk()

frm = Frame(root)
frm.pack(side=BOTTOM, padx = 15, pady = 15)

lbl = Label(root)
lbl.pack()

#Botões
btmVideo = Button(frm, text='Video', command=lambda: detecta_video())
btmVideo.pack(side=tk.LEFT)

btmImagem = Button(frm, text='Imagem', command=lambda: detecta_imagem())
btmImagem.pack(side=tk.LEFT)

btmSair = Button(frm, text='Sair', command=lambda: exit())
btmSair.pack(side=tk.LEFT)

root.title("Menu")
root.geometry('300x75+600+300')

root.mainloop()