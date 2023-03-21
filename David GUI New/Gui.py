import customtkinter
from tkinter import *
import subprocess
from threading import Thread
import threading

# Global Variables


#Interface Stuffs
root = customtkinter.CTk()
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")
root.title("Hand Gestural Recognition System")
root.geometry = ("1024x720")
myFrame = Frame(root, width=1024, height=720)
myFrame.pack()

def test():
    print("Null")

def DTW():
        subprocess.call(f'python C:\\Users\\david\\Desktop\\SeniorDesign-main\\SeniorDesign-main\\DTWpipe\\main.py', shell=True)



if __name__ == "__main__":

    dtwThread = Thread(target = DTW)
    dtwThread.setDaemon(True)
    
    btn = Button(root, text = 'START TRANSLATING', bd = '5', command = dtwThread.start)
    btn.pack(side = 'top')
    root.mainloop()
