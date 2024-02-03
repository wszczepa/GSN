import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
import cv2
import tkinter
from tkinter import filedialog
from tkinter import*
import tkinter as tk
import cv2
from ultralytics import YOLO

model_yolo = YOLO("F:/miniconda/models/best.pt")
state_name = [f.name for f in os.scandir('F:/miniconda/new_dataset2/new plates/train') if f.is_dir()]


def showimg(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

def classify(img):
    
    resize = tf.image.resize(img, (200,200))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    text_var.set(f"Clissify result: {state_name[(np.argmax(yhat, axis=1)).item()]}, Certainty Level: {yhat[0][(np.argmax(yhat)).item()]:.3f}")
def add_file():
        global model_name
        global img_name
        global img_rgb 
        global model
        file_name = filedialog.askopenfilename(initialdir='F:\miniconda',
                                               title="Select File",
                                               filetypes=(("All files", "*.*"),))
        
        
        tmp1 = file_name.split("/")[-1]
        tmp2 = tmp1.split(".")

        if file_name!="" and tmp2[1] == "png" or tmp2[1] == "jpg":
            img_name = tmp2[0]
            addAppinfo = tkinter.Label()
            addAppinfo = tkinter.Label(text="                                                                                                                        ")
            addAppinfo.place(x=120, y=25)
            addAppinfo = tkinter.Label(text=f"Plik {tmp2[0]}.{tmp2[1]} wczytany")
            addAppinfo.place(x=120,y=25)
        elif file_name!="" and tmp2[1] == "h5":
            model_name = tmp2[0]
            addAppinfo2 = tkinter.Label()
            addAppinfo2 = tkinter.Label(text="                                                                                                                        ")
            addAppinfo2.place(x=120, y=65)
            addAppinfo2 = tkinter.Label(text=f"Plik {tmp2[0]}.{tmp2[1]} wczytany")
            addAppinfo2.place(x=120,y=65)
        if tmp2[1] == "png" or tmp2[1] == "jpg":
            img_rgb = cv2.imread(os.path.join(file_name))
            #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif tmp2[1] == "h5":
            model = tf.keras.models.load_model(file_name)



def cut_img(img):
    result = model_yolo.predict(img)[0]
    for i, xyxy in enumerate(result.boxes.xyxy):
        x1, y1, x2, y2 = map(int, xyxy[:4])

        roi = img[y1-5:y2+5, x1-5:x2+5]

        cv2.imwrite(f"F:/miniconda/pictures/plate_photo/{img_name}_cut.jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 100,])

        
        
        
def cut_and_clissify(img):

    cut_img(img)
    img_cut = cv2.imread(f"F:/miniconda/pictures/plate_photo/{img_name}_cut.jpg")
    classify(img_cut)


def show_cut_img(img_name):
    img_cut = cv2.imread(f"F:/miniconda/pictures/plate_photo/{img_name}_cut.jpg")
    img_cut_rgb = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)
    plt.imshow(img_cut_rgb)
    plt.show()

okno =Tk()
okno.geometry('480x480')
okno.resizable(False,False)
okno.wm_title("Projekt_sieci")

insert = Entry(okno,width=11)
insert2 = Entry(okno,width=12)
openFile = tkinter.Button(okno, text='Load img',command=lambda:add_file())
button = tkinter.Button(okno, text='Load model',command=lambda:add_file())
button2 = tkinter.Button(okno, text='Show img',command=lambda:showimg(img_rgb))
button3 = tkinter.Button(okno, text='Cut img',command=lambda:cut_img(img_rgb))
button6 = tkinter.Button(okno, text='Show cut img',command=lambda:show_cut_img(img_name))
button4 = tkinter.Button(okno, text='Classify',command=lambda:classify(img_rgb))
button5 = tkinter.Button(okno, text='Cut and classify',command=lambda:cut_and_clissify(img_rgb))


#labele
text_var = tk.StringVar()
text_var.set(" ")  # Ustawienie początkowego tekstu
time_size = tkinter.Label(okno, textvariable=text_var)
time_size.pack()
time_size.place(x=110,y=340)

#inserty
# insert.pack()
# insert.place(x=20,y=220)
# insert2.pack()
# insert2.place(x=140,y=220)
#przyciski
openFile.pack()
openFile.place(x=20,y=20)
button.pack()
button.place(x=20,y=60)
button2.pack()
button2.place(x=20,y=100)
button3.pack()
button3.place(x=20,y=140)
button6.pack()
button6.place(x=20,y=180)
button4.pack()
button4.place(x=20,y=220)
button5.pack()
button5.place(x=20,y=260)



okno.mainloop()






