import sys
import os
from pathlib import Path
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout, QFileDialog, QComboBox
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtCore import *
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

import omp


def cv2QImage(image: np.ndarray) -> QImage:
    image_h, image_w = image.shape
    image_bytes_per_line = image_w

    pixel_len = image_h * image_w

    qImage = QImage(image.data, image_w, image_h, image_bytes_per_line, QImage.Format.Format_Grayscale8)

    return qImage

class MainWindow(QMainWindow):
    button_styles = '''
                        font-weight: bold;
                        height: 48px;
                        width: 10px;
                    '''

    avail_algorithms = ["OMP"]
    fname_image = ""
    image_w = 0
    image_h = 0
    image_compressed = np.ndarray((0, 0))
    source_image = np.ndarray((0, 0))

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
        
        combobox = QComboBox()
        for alg in self.avail_algorithms:
            combobox.addItem(alg)
        combobox.setStyleSheet('''
                                height: 40px;
                            ''')

        button = QPushButton("Выбрать изображение")
        button.setStyleSheet(self.button_styles)
        #button.setFixedSize(QSize(150, 60))
        button.clicked.connect(self.openFileDialog)

        layout.addWidget(button)
        layout.addWidget(combobox)

        widget = QWidget()
        widget.setLayout(layout)
        
        self.setCentralWidget(widget)

    def runOMP(self):
        self.image_compressed = omp.omp(self.fname_image, omp.dct(self.image_w), self.image_w // 2, 50)

        lb2 = QLabel(self)
        cv2.imwrite("compressed.png", self.image_compressed)
        lb2.setPixmap(QPixmap(cv2QImage(cv2.imread("compressed.png", cv2.IMREAD_GRAYSCALE))))

        lb = QLabel(self)
        lb.setPixmap(QPixmap(cv2QImage(self.source_image)))

        layout = QVBoxLayout()

        combobox = QComboBox()
        for alg in self.avail_algorithms:
            combobox.addItem(alg)

        button = QPushButton("Выбрать изображение")
        button.setStyleSheet(self.button_styles)
        button.clicked.connect(self.openFileDialog)
        
        run_button = QPushButton("Запуск")
        run_button.setStyleSheet(self.button_styles)
        run_button.clicked.connect(self.runOMP)


        layout.addWidget(button)
        layout.addWidget(combobox)
        layout.addWidget(run_button)

        layout.addWidget(lb)
        layout.addWidget(lb2)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


    def openFileDialog(self):
        fname_E, selFilter = QFileDialog.getOpenFileName()
        self.fname_image = fname_E

        image = cv2.imread(fname_E, cv2.IMREAD_GRAYSCALE)
        self.source_image = image
        self.image_h, self.image_w = image.shape
        image_bytes_per_line = self.image_w

        pixel_len = self.image_h * self.image_w

        lb = QLabel(self)

        #lb.resize(300, int((image_h * 300) / image_w))
        #lb.setPixmap(pixmap.scaled(lb.size()))
        lb.setPixmap(QPixmap(cv2QImage(image)))
        
        layout = QVBoxLayout()
        
        combobox = QComboBox()
        for alg in self.avail_algorithms:
            combobox.addItem(alg)

        button = QPushButton("Выбрать изображение")
        button.setStyleSheet(self.button_styles)
        button.clicked.connect(self.openFileDialog)
        
        run_button = QPushButton("Запуск")
        run_button.setStyleSheet(self.button_styles)
        run_button.clicked.connect(self.runOMP)


        layout.addWidget(button)
        layout.addWidget(combobox)
        layout.addWidget(run_button)

        layout.addWidget(lb)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        #print(pixel_len)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()
