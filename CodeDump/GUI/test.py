import cv2
import mediapipe as mp
import csv
from PIL import Image
from numpy import asarray
from vecSave import vecSave
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from glob import glob
from tkvideo import tkvideo
import os
import cv2
import imutils
import logging
import numpy as np
import threading
import sys

from mediPipe import mediaPipe

global cap
global img

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# global count
# count = 0
# global vec_list
# vec_list = []

imageCounter = 0
fontSize = 12
pausePlayText = 1
pBool = 1
cameraFlag = 1
translatedText = "Translated Text Displayed Here!"

# Initializing Live Recording
def initalizeLive():
    global img, cameraFlag, cap
    if cameraFlag == 1:
        cap = cv2.VideoCapture(0)
        cameraFlag = cameraFlag - 1
        liveTranslation()

# def whileLive():
#     global img, cameraFlag, cap
#     if cameraFlag == 0:
#         # liveTranslation()
#     else:
#         cap.release()
#         cv2.destroyAllWindows()
#         newImage = ImageTk.PhotoImage(Image.open("default.png"))
#         imageLabel.configure(image=newImage)
#         imageLabel.image=newImage
#         cameraFlag = 1

# For testing purposes only
def liveTranslation():
    global img
    success, frame = cap.read()
    mediaPipe(success, frame)
    # img = Image.fromarray(cv2image) #previmg
    # imgtk = ImageTk.PhotoImage(image=img)
    # imageLabel.imgtk = imgtk
    # imageLabel.configure(image=imgtk)
    # imageLabel.after(pBool, whileLive)


# Changing text for Pause/Play Button
def pausePlay():
    global pBool
    global pausePlayText
    pausePlayText = not(pausePlayText)
    if pausePlayText:
        logging.info("Live Recording - Recording")
        pausePlayButton.config(text="STOP")
        pBool = 1
    else:
        logging.info("Live Recording - Paused")
        pausePlayButton.config(text="START")
        pBool = 0

# def record():
#     writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (300,300))
#     while True:
#         ret,frame= cap.read()
#         writer.write(frame)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#
#     cap.release()
#     writer.release()
#     cv2.destroyAllWindows()


def stopRecording():
    global cameraFlag
    cameraFlag = cameraFlag - 1
    # whileLive()


# Save the translated text
def save():
    global img, imageCounter
    if (len(sys.argv) < 2):
        filepath = "myImage" + str(imageCounter)
        imageCounter = imageCounter + 1
    else:
        filepath = sys.argv[1]

    print ("Output file to: " + filepath)
    img.save(filepath + ".png")


# Translate the ASL media into text
def translate():
        textBox.delete("1.0", "end")
        textBox.insert('1.0', "Once the user loads an image/video to this program it will display the text detected in this box!")

# Add an image to the program
def addImage():
    # Get the new images file path
    newImageFilePath = filedialog.askopenfilename(initialdir = "/", title = "Select Media", filetypes = (("Image Files", "*.png*"), ("Image Files", "*.jpg*"), ("Video Files", "*.mp4*")))
    # Saving the image name to imageName variable
    imageName = os.path.basename(newImageFilePath)

    # Checking if video or image
    if ".mp4" in imageName:
        player = tkvideo(newImageFilePath, imageLabel, size = (300,300))
        player.play()
    else:
        # Testing to make sure the image name is correct
        print(imageName)
        # Resizing the image
        newResizedImage = Image.open(newImageFilePath)
        newResizedImage = newResizedImage.resize((300, 300))
        newResizedImage.save(imageName)
        # Opening the image
        newImage = ImageTk.PhotoImage(Image.open(imageName))
        imageLabel.configure(image=newImage)
        imageLabel.image=newImage


# Delcaring and Initializing
#translatedText = "Translated Text Displayed Here!"

# DEBUG information
level = logging.DEBUG
fmt = '[%(levelname)s] %(asctime)s - %(message)s'
logging.basicConfig(level=level, format=fmt)
logging.info("Program Start.")

# Creating the window
windows = tk.Tk()
windows.title("Senior Design Project")
windows.geometry("900x600")
windows.resizable(False, False)
verticalScroll = Scrollbar(windows, orient='vertical')
verticalScroll.pack(side = RIGHT, fill = 'y')

# Tab Control
tabs = ttk.Notebook(windows)

# Translation tab
translateTab = ttk.Frame(tabs)
tabs.add(translateTab, text='Translate')

# Title
Label(translateTab, text= "Hand Gestural Recognition System",font=('Arial', 26), pady=(20)).pack()

# Recent Translations tab
recentTranslationsTab = ttk.Frame(tabs)
tabs.add(recentTranslationsTab, text='Recent Translations')

# Saved Translations tab
savedTranslationsTab = ttk.Frame(tabs)
tabs.add(savedTranslationsTab, text='Saved Translations')

# Options Translations tab
optionsTab = ttk.Frame(tabs)
tabs.add(optionsTab, text='Options')

# Help tab
helpTab = ttk.Frame(tabs)
tabs.add(helpTab, text='Help')

tabs.pack(expand=True, fill='both')

# Initializing Frame
frame = Frame(windows)
frame.pack()

# TRANSLATION TAB
# Initializing default image
signImage = ImageTk.PhotoImage(Image.open("default.png"))
imageLabel = Label(translateTab, image = signImage)
imageLabel.pack(expand=True, fill="both")

# text box
textBox = Text(translateTab, height = 8, width = 75, font=('Arial', fontSize))
textBox.pack(expand = False)
textBox.insert('end', translatedText)
#textBox.config(state='disabled')


# Live translation Button
liveButton = Button(translateTab, text = "REC", font=('Arial', 16), fg="red", height=1, width=5, command = initalizeLive)
liveButton.place(x=25, y=106)

# Live translation Button
pausePlayButton = Button(translateTab, text = "STOP", font=('Arial', 16), fg="blue", height=1, width=7, command = pausePlay)
pausePlayButton.place(x=100, y=106)

# stopRecording Button
stopRecordingButton = Button(translateTab, text = "STOP REC", font=('Arial', 16), fg="red", height=1, width=10, command = stopRecording)
stopRecordingButton.place(x=25, y=156)

# addMedia Button
addMediaButton = Button(translateTab, text = "Add Media", font=('Arial', 16), fg="blue", height=1, width=10, command = addImage)
addMediaButton.place(x=25, y=206)

# translate Button
translateButton = Button(translateTab, text = "Translate", font=('Arial', 16), fg="blue", height=1, width=10, command = translate)
translateButton.place(x=25, y=256)

# save translation Button
saveTranslationButton = Button(translateTab, text = "Save", font=('Arial', 16), fg="blue", height=1, width=10, command = save)
saveTranslationButton.place(x=25, y=306)
###########################################################################################################################



# OPTIONS TAB
###########################################################################################################################

# HELP TAB
helpText = tk.Label(helpTab, text='There is no helping you', font=('Arial', 26))
helpText.grid(row=1,column=1)
###########################################################################################################################

windows.mainloop()
