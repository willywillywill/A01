import cv2
import pickle
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from functions import sigmoid, softmax

def get_img(img):
    def predict(x):
        network=pickle.load(open('A01/number.pickle','rb'))
        W1=network["W1"]
        b1=network["b1"]
        W2=network["W2"]
        b2=network["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    def normalize(img):
        mean=np.mean(img)
        var=np.mean(np.square(img-mean))
        img=(img-mean)/np.sqrt(var)
        
        return img
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    img = np.array(img).flatten().astype(np.float32)
    img = normalize(img)

    y = np.argmax(predict(img))

    return y 

def video_loop():    
    _, img = cam.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    y = get_img(img)

    L1.configure(text=str(y))
        
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_L.imgtk = imgtk
    video_L.configure(image=imgtk)
    video_L.after(10, video_loop)

root = tk.Tk()
root.title("cam")
root.geometry("900x500")
video_L = tk.Label(root)
video_L.pack(side=tk.LEFT)

cam = cv2.VideoCapture(0)
L1 = tk.Label(root,width=5,height=5,font=(80,80),text="")
L1.pack(side=tk.LEFT)

video_loop()

root.mainloop()