import PyQt5.QtWidgets
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap
import sys
import cv2
import numpy as np
from keras.models import load_model

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('GUI.ui', self)

        self.setWindowTitle("Breast cancer prediction")

        self.setWindowIcon(QIcon(r"C:\Users\tomer\Downloads\finals app\icon.jpg"))

        self.button_load_image = self.findChild(QtWidgets.QPushButton, 'ButtonLoadImage')
        self.button_load_image.clicked.connect(self.load_image)

        self.button_predict = self.findChild(QtWidgets.QPushButton, 'ButtonPredict')
        self.button_predict.clicked.connect(self.predict)

        self.label_title = self.findChild(QtWidgets.QLabel, 'LabelTitle')
        self.label_answer = self.findChild(QtWidgets.QLabel, 'LabelAnswer')
        self.label_answer2 = self.findChild(QtWidgets.QLabel, 'LabelAnswer2')
        self.label_back = self.findChild(QtWidgets.QLabel, 'label_back')

        self.label_photo = self.findChild(QtWidgets.QLabel, 'label_photo')
        self.setFixedWidth(977)
        self.setFixedHeight(892)
        self.path = ""
        self.img = ""

    def load_image(self):
        self.label_answer.setText("")
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open File",
                                                      r"C:\Users\tomer\Downloads",
                                                      "Images (*.png)")
        if not fname[0]:
            return

        path = fname[0]
        self.path = path
        self.pixmap = QPixmap(path)
        self.label_photo.setPixmap(self.pixmap)
        PyQt5.QtWidgets.QApplication.processEvents()

    def predict(self):
        if not self.path:
            self.label_answer.setText("Load an image first.")
            self.label_answer.setStyleSheet("color: rgb(0, 0, 0);")
            return

        img = cv2.imread(self.path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        y_pred = model.predict(img)
        prediction = y_pred[0][0]
        y2_pred = model2.predict(img)
        prediction2 = y2_pred[0][0]
        print(prediction)
        print(prediction2)
        if prediction > 0.5:
            self.label_answer.setText(f"model1: possible positive, The probability of a tumor is: {100*prediction:.2f}%")
            self.label_answer.setStyleSheet("color: rgb(250, 0, 0);")
        else:
            self.label_answer.setText(f"model1: possible negative, The probability of a tumor is: {100*prediction:.2f}%")
            self.label_answer.setStyleSheet("color: rgb(0, 170, 0);")

        if prediction2 > 0.5:
            self.label_answer2.setText(f"model2: possible positive, The probability of a tumor is: {100*prediction2:.2f}%")
            self.label_answer2.setStyleSheet("color: rgb(250, 0, 0);")
        else:
            self.label_answer2.setText(f"model2: possible negative, The probability of a tumor is: {100*prediction2:.2f}%")
            self.label_answer2.setStyleSheet("color: rgb(0, 170, 0);")

# Load the pre-trained model


model = load_model(r'C:\Users\tomer\Downloads\finals app\inception_hist\my_densenet_model.h5')
model2 = load_model(r'C:\Users\tomer\Downloads\finals app\inception_hist\my_inception_model.h5')
app = QtWidgets.QApplication(sys.argv)
widget = QtWidgets.QStackedWidget()
window = Ui()
widget.addWidget(window)
widget.show()
app.exec_()
