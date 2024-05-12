import logging
import os

import tensorflow as tf
from PyQt6 import QtCore, uic
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QGridLayout, QWidget, QVBoxLayout, QLabel
from deepface import DeepFace
from qt_async_threads import QtAsyncRunner

form_class = uic.loadUiType("main_window.ui")[0]  # Load the UI


class MyWindowClass(QMainWindow, form_class):
    def __init__(self, parent=None):
        self.runner = QtAsyncRunner()
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.targetFaceFileChooseButton.clicked.connect(self.targetFaceFileChooseButton_clicked)
        self.directoryChooseButton.clicked.connect(self.directoryChooseButton_clicked)
        self.startAnalyzeButton.clicked.connect(self.runner.to_sync(self.startAnalyzeButton_clicked))

    def targetFaceFileChooseButton_clicked(self):
        # options = QFileDialog.options(self)
        # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*)", options=options)
        fileName, _ = QFileDialog.getOpenFileName()
        if fileName:
            pixmap = QPixmap(fileName).scaled(self.targetFaceImageLabel.width(),  self.targetFaceImageLabel.height())
            self.targetFaceImageLabel.setPixmap(pixmap)
            self.targetFaceImageLabel.setText(fileName)

    def directoryChooseButton_clicked(self):
        directoryPath = QFileDialog.getExistingDirectory()
        if directoryPath:
            self.directoryPathLabel.setText(directoryPath)
            self.directoryPathLabel.adjustSize()

    async def startAnalyzeButton_clicked(self, checked: bool = False):
        directory_path = self.directoryPathLabel.text()
        target_face_image_path = self.targetFaceImageLabel.text()

        gpus = tf.config.list_physical_devices('GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

        for (dirpath, dirnames, filenames) in os.walk(directory_path):
            if dirpath != directory_path:
                continue
            total = len(filenames)
            self.analyzeProgressBar.setMaximum(total)

            resultGridLayout = QGridLayout()
            widget = QWidget()
            widget.setLayout(resultGridLayout)
            self.foundFaceImgsScrollArea.setWidget(widget)

            found = 0

            for (i, filename) in enumerate(filenames):
                img2_path = os.path.join(dirpath, filename)

                # Вызов функции DeepFace для сравнения лиц
                try:
                    #DeepFace.verify(target_face_image_path, img2_path, distance_metric="euclidean_l2", expand_percentage=0,enforce_detection=True, model_name="Facenet512",detector_backend="retinaface")
                    result = await self.runner.run(
                        DeepFace.verify, target_face_image_path, img2_path, "Facenet512", "retinaface", "euclidean_l2"
                    )
                except Exception as e:
                    logging.exception(e)
                    continue

                analyzed = i + 1
                self.analyzeProgressBar.setValue(analyzed)
                self.analyzeProgressLabel.setText(f"Обработано {analyzed}/{total} изображений")
                self.analyzeProgressLabel.adjustSize()

                # Вывод результата
                if result["verified"]:

                    await self.handleFoundFace(found, img2_path, resultGridLayout, result)

                    found += 1

    async def handleFoundFace(
            self, foundCount: int, foundFaceImgPath: str, resultGridLayout: QGridLayout, result: dict
    ):
        verticalLayout = QVBoxLayout()
        foundImglabel = QLabel()
        foundImglabel.height = 400
        foundImglabel.width = 300
        foundImglabel.setPixmap(QPixmap(foundFaceImgPath).scaled(foundImglabel.width, foundImglabel.height))

        foundImgPathLabel = QLabel()
        foundImgPathLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        foundImgPathLabel.setText(foundFaceImgPath)
        foundImgPathLabel.setWordWrap(True)
        foundImgPathLabel.adjustSize()

        verticalLayout.addWidget(foundImglabel)
        verticalLayout.addWidget(foundImgPathLabel)

        rowNumber = int(foundCount / 4)
        colNumber = foundCount % 4
        resultGridLayout.addLayout(verticalLayout, rowNumber, colNumber)

        print(
            f"Лица на изображениях {foundFaceImgPath} похожи, Уровень похожести: {result['distance']}, result: {result}"
        )