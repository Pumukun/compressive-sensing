import sys
import os
from pathlib import Path
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout, QFileDialog, QComboBox
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtCore import *
from PyQt6.QtGui import QPixmap
from PIL import Image



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
        
        combobox = QComboBox()
        combobox.addItem('One')
        combobox.addItem('Two')
        combobox.addItem('Three')
        combobox.addItem('Four')
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

   

    def openFileDialog(self):
        fname_E, selFilter = QFileDialog.getOpenFileName()
        image = Image.open(fname_E)
        pixel_len = image.width * image.height

        image_data = image.getdata()     

        lb = QLabel(self)
        pixmap = QPixmap(fname_E)
        lb.resize(300, int((image.height * 300) / image.width))
        lb.setPixmap(pixmap.scaled(lb.size()))

        
        layout = QVBoxLayout()
        
        combobox = QComboBox()
        combobox.addItem('One')
        combobox.addItem('Two')
        combobox.addItem('Three')
        combobox.addItem('Four')

        button = QPushButton("Выбрать изображение")
        button.setStyleSheet(self.button_styles)
        button.clicked.connect(self.openFileDialog)
        
        layout.addWidget(button)
        layout.addWidget(combobox)

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