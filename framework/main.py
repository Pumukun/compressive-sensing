import sys
import os
from pathlib import Path
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout, QFileDialog, QComboBox
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtCore import *
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
import cv2
import omp as o



class MainWindow(QMainWindow):
    button_styles = '''
                        font-weight: bold;
                        height: 48px;
                        width: 10px;
                    '''


    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("ашалеть")
        self.setStyleSheet('''
                            font-size: 16px;
                            font-family: "arial";
                            ''') 
        self.resize(800, 600)

        layout = QVBoxLayout()
        
        button = QPushButton("Выбрать изображение")
        button.setStyleSheet(self.button_styles)
        #button.setFixedSize(QSize(150, 60))
        button.clicked.connect(self.openFileDialog)

        layout.addWidget(button)

        widget = QWidget()
        widget.setLayout(layout)
        
        self.setCentralWidget(widget)

   

    def openFileDialog(self):
        fname_E, selFilter = QFileDialog.getOpenFileName()
        image = Image.open(fname_E)
        pixel_len = image.width * image.height

        image_data = image.getdata()

        lb_before = QLabel(self)
        lb_after = QLabel(self)

        lb_before.resize(300, int((image.height * 300) / image.width))
        pixmap_before = QPixmap(fname_E)
        lb_before.setPixmap(pixmap_before)

        m = 128
        k = 10

        rec = o.omp(fname_E, o.dct(256), m, k)
        cv2.imwrite(f"tmp/lena_omp.png", rec)

        fname_omp = "tmp/lena_omp.png"
        cvImg = cv2.imread(fname_omp)
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)



        lb_after.resize(300, int((image.height * 300) / image.width))
        pixmap_after = QPixmap(qImg)
        lb_after.setPixmap(pixmap_after)

        
        layout = QVBoxLayout()
        

        button = QPushButton("Выбрать изображение")
        button.setStyleSheet(self.button_styles)
        button.clicked.connect(self.openFileDialog)
        
        layout.addWidget(button)
        layout.addWidget(lb_before)
        layout.addWidget(lb_after)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        #print(pixel_len)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()