'''
ITU - Computer Vision - HW2
Bunyamin Kurt - 150140145
'''
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5 import QtWidgets

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def main():
    app = QApplication(sys.argv)
    ex = UserInterface()
    sys.exit(app.exec_())

class UserInterface(QMainWindow):

    def __init__(self):
        super().__init__()
        self.inputImg = ""
        self.resultImg = ""
        self.initUI()

    def initUI(self):

        # Create Open Input Image Action
        openInputAct = QAction('&Open', self)
        openInputAct.setStatusTip('Open')
        openInputAct.triggered.connect(self.openImage)

        # Create Target Input Image Action
        openTargetAct = QAction('&Save Result Omage', self)
        openTargetAct.setStatusTip('Save Result Image')
        openTargetAct.triggered.connect(self.saveImage)

        # Create Exit Action
        exitAct = QAction('&Exit', self)
        exitAct.setStatusTip('Exit')
        exitAct.triggered.connect(qApp.quit)

        # Average Filter Menu
        averageFiltersMenu = QMenu('Average Filters', self)
        impAct = QAction('3x3', self)
        impAct.triggered.connect(lambda: self.averageFilter(3))
        impAct2 = QAction('5x5', self)
        impAct2.triggered.connect(lambda: self.averageFilter(5))
        impAct3 = QAction('7x7', self)
        impAct3.triggered.connect(lambda: self.averageFilter(7))
        impAct4 = QAction('9x9', self)
        impAct4.triggered.connect(lambda: self.averageFilter(9))
        impAct5 = QAction('11x11', self)
        impAct5.triggered.connect(lambda: self.averageFilter(11))
        impAct6 = QAction('13x13', self)
        impAct6.triggered.connect(lambda: self.averageFilter(13))
        impAct7 = QAction('15x15', self)
        impAct7.triggered.connect(lambda: self.averageFilter(15))
        averageFiltersMenu.addAction(impAct)
        averageFiltersMenu.addAction(impAct2)
        averageFiltersMenu.addAction(impAct3)
        averageFiltersMenu.addAction(impAct4)
        averageFiltersMenu.addAction(impAct5)
        averageFiltersMenu.addAction(impAct6)
        averageFiltersMenu.addAction(impAct7)

        # Gaussian Filter Menu
        gaussianFiltersMenu = QMenu('Gaussian Filters', self)
        impAct = QAction('3x3', self)
        impAct.triggered.connect(lambda: self.gaussianFilter(3))
        impAct2 = QAction('5x5', self)
        impAct2.triggered.connect(lambda: self.gaussianFilter(5))
        impAct3 = QAction('7x7', self)
        impAct3.triggered.connect(lambda: self.gaussianFilter(7))
        impAct4 = QAction('9x9', self)
        impAct4.triggered.connect(lambda: self.gaussianFilter(9))
        impAct5 = QAction('11x11', self)
        impAct5.triggered.connect(lambda: self.gaussianFilter(11))
        impAct6 = QAction('13x13', self)
        impAct6.triggered.connect(lambda: self.gaussianFilter(13))
        impAct7 = QAction('15x15', self)
        impAct7.triggered.connect(lambda: self.gaussianFilter(15))
        gaussianFiltersMenu.addAction(impAct)
        gaussianFiltersMenu.addAction(impAct2)
        gaussianFiltersMenu.addAction(impAct3)
        gaussianFiltersMenu.addAction(impAct4)
        gaussianFiltersMenu.addAction(impAct5)
        gaussianFiltersMenu.addAction(impAct6)
        gaussianFiltersMenu.addAction(impAct7)

        # Median Filter Menu
        medianFiltersMenu = QMenu('Median Filters', self)
        impAct = QAction('3x3', self)
        impAct.triggered.connect(lambda: self.medianFilter(3))
        impAct2 = QAction('5x5', self)
        impAct2.triggered.connect(lambda: self.medianFilter(5))
        impAct3 = QAction('7x7', self)
        impAct3.triggered.connect(lambda: self.medianFilter(7))
        impAct4 = QAction('9x9', self)
        impAct4.triggered.connect(lambda: self.medianFilter(9))
        impAct5 = QAction('11x11', self)
        impAct5.triggered.connect(lambda: self.medianFilter(11))
        impAct6 = QAction('13x13', self)
        impAct6.triggered.connect(lambda: self.medianFilter(13))
        impAct7 = QAction('15x15', self)
        impAct7.triggered.connect(lambda: self.medianFilter(15))
        medianFiltersMenu.addAction(impAct)
        medianFiltersMenu.addAction(impAct2)
        medianFiltersMenu.addAction(impAct3)
        medianFiltersMenu.addAction(impAct4)
        medianFiltersMenu.addAction(impAct5)
        medianFiltersMenu.addAction(impAct6)
        medianFiltersMenu.addAction(impAct7)

        # Rotate Menu
        rotateMenu = QMenu('Rotate', self)
        impAct = QAction('Rotate 10 Degree Right', self)
        impAct.triggered.connect(lambda: self.rotate10("R"))
        impAct2 = QAction('Rotate 10 Degree Left', self)
        impAct2.triggered.connect(lambda: self.rotate10("L"))
        rotateMenu.addAction(impAct)
        rotateMenu.addAction(impAct2)

        # Scale Menu
        scaleMenu = QMenu('Scale', self)
        impAct = QAction('2x', self)
        impAct.triggered.connect(lambda: self.scale("2x"))
        impAct2 = QAction('1/2x', self)
        impAct2.triggered.connect(lambda: self.scale("1/2x"))
        scaleMenu.addAction(impAct)
        scaleMenu.addAction(impAct2)

        # Translate Menu
        translateMenu = QMenu('Translate', self)
        impAct = QAction('Right', self)
        impAct.triggered.connect(lambda: self.translate("R"))
        impAct2 = QAction('Left', self)
        impAct2.triggered.connect(lambda: self.translate("L"))
        translateMenu.addAction(impAct)
        translateMenu.addAction(impAct2)

        # Create menuBar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openInputAct)
        fileMenu.addAction(openTargetAct)
        fileMenu.addAction(exitAct)

        # Add filter menu to main menubar
        filterMenu = menubar.addMenu('&Filters')
        filterMenu.addMenu(averageFiltersMenu)
        filterMenu.addMenu(gaussianFiltersMenu)
        filterMenu.addMenu(medianFiltersMenu)

        # Add geometric transformation menu to main menubar
        geometricTransformsrMenu = menubar.addMenu('&Geometric Transforms')
        geometricTransformsrMenu.addMenu(rotateMenu)
        geometricTransformsrMenu.addMenu(scaleMenu)
        geometricTransformsrMenu.addMenu(translateMenu)

        # Create main widget
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create Horizantal Box Layout
        self.vboxH = QHBoxLayout()
        self.vbox1 = QVBoxLayout()

        # Add input label
        title = QLabel('Input Image')
        self.vbox1.addWidget(title)

        # Create vertical box layout
        self.vbox2 = QVBoxLayout()

        # Add target label
        title = QLabel('Result Image')
        self.vbox2.addWidget(title)

        # Add vertical box layout into Horizantal box layout
        self.vboxH.addLayout(self.vbox1)
        self.vboxH.addLayout(self.vbox2)
        wid.setLayout(self.vboxH)

        self.setGeometry(200, 200, 400, 300)
        self.setWindowTitle('Histogram Equalization')
        self.show()

    def openImage(self):
        # Open file dialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.inputImg, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)

        if self.inputImg:
            # Insert image into first vertical box layout
            l1 = QLabel()
            l1.setPixmap(QPixmap(self.inputImg))
            self.vbox1.addWidget(l1)

    def saveImage(self):
        # Open file dialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Png Files (*.png)", options=options)

        if fileName and self.resultImg:
            print(fileName)
        else:
            QMessageBox.about(self, "Error", "Result image not found or you did not enter a proper filename")

    def rotate10(self,direction):
        # Show Error message if input image did not select
        if not self.inputImg:
            QMessageBox.about(self, "Error", "Input image not found!")
            return

        # Read selected image and seperate red, green and blue
        img = mpimg.imread(self.inputImg)
        r = img[:,:,0];
        g = img[:,:,1];
        b = img[:,:,2];

        # Get image height and width
        height = r.shape[0]
        width = r.shape[1]

        # Calculate angle according to right or left
        if direction == "R":
            angle = np.pi/18
        else:
            angle = -1 * np.pi/18

        # Calculate center of image
        centerx = width // 2
        centery = height // 2

        # Create new image same height and width
        newImage = np.zeros((img.shape[0],img.shape[1],3), np.float64)
        # Iterate over image
        for row in range(0, height):
            for col in range(0, width):
                # Calculate new coordinates
                rot_mat = np.asarray([[np.cos(angle), np.sin(angle)], [-1 * np.sin(angle), np.cos(angle)]])
                coord = np.matmul(rot_mat, np.asarray([col - centerx, row - centery]))
                coord += (centerx, centery)
                # If new coordinates is within the image borders
                if int(coord[0]) >= 1 and int(coord[1]) >= 1 and int(coord[1]) < height and int(coord[0]) < width:
                    # Insert into new image to new values
                    newImage[...,0][row,col] = r[int(coord[1]),int(coord[0])]
                    newImage[...,1][row,col] = g[int(coord[1]),int(coord[0])]
                    newImage[...,2][row,col] = b[int(coord[1]),int(coord[0])]

        # Save result image
        plt.clf()
        plt.imshow(newImage)
        plt.show()
        '''plt.savefig("resultPic.png")

        # Insert result image into third vertical box layout
        l2 = QLabel()
        l2.setPixmap(QPixmap("resultPic.png"))
        self.vbox2.addWidget(l2)'''

    def scale(self, factor):
        # Show Error message if input image did not select
        if not self.inputImg:
            QMessageBox.about(self, "Error", "Input image not found!")
            return

        # Read selected image and seperate red, green and blue
        img = mpimg.imread(self.inputImg)
        r = img[:,:,0];
        g = img[:,:,1];
        b = img[:,:,2];

        # Get image height and width
        height = r.shape[0]
        width = r.shape[1]

        # Calculate center of image
        centerx = width // 2
        centery = height // 2

        # Create new image same height and width
        newImage = np.zeros((img.shape[0],img.shape[1],3), np.float64)
        # Iterate over image
        for row in range(0, height):
            for col in range(0, width):
                # If user select 2x then sacale 2x
                if factor == "2x":
                    scale_mat = np.eye(2) * 2
                else:
                    scale_mat = np.eye(2) * 0.5
                # Calculate new coordinates
                scale_mat = np.linalg.inv(scale_mat)
                coord = np.matmul(scale_mat, np.asarray([col - centerx, row - centery]))
                coord += (centerx, centery)
                # If new coordinates is within the image borders
                if int(coord[0]) >= 1 and int(coord[1]) >= 1 and int(coord[1]) < height and int(coord[0]) < width:
                    # Insert into new image to new values
                    newImage[...,0][row,col] = r[int(coord[1]),int(coord[0])]
                    newImage[...,1][row,col] = g[int(coord[1]),int(coord[0])]
                    newImage[...,2][row,col] = b[int(coord[1]),int(coord[0])]
        plt.clf()
        plt.imshow(newImage)
        plt.show()

    def translate(self, rotation):
        # Show Error message if input image did not select
        if not self.inputImg:
            QMessageBox.about(self, "Error", "Input image not found!")
            return

        # Read selected image and seperate red, green and blue
        img = mpimg.imread(self.inputImg)
        r = img[:,:,0];
        g = img[:,:,1];
        b = img[:,:,2];

        # Get image height and width
        height = r.shape[0]
        width = r.shape[1]

        # Create new image same height and width
        newImage = np.zeros((img.shape[0],img.shape[1],3), np.float64)
        # Iterate over image
        for row in range(0, height):
            for col in range(0, width):
                # If user select left and we can go 100 pixels left
                if rotation == "L" and not col + 100 > width - 1:
                    # Insert into new image 100 pixels left
                    newImage[...,0][row][col] = r[row][col + 100]
                    newImage[...,1][row][col] = g[row][col + 100]
                    newImage[...,2][row][col] = b[row][col + 100]

                # If user select right and we can go 100 pixels right
                if rotation == "R" and not col - 100 < 0:
                    # Insert into new image 100 pixels right
                    newImage[...,0][row][col] = r[row][col - 100]
                    newImage[...,1][row][col] = g[row][col - 100]
                    newImage[...,2][row][col] = b[row][col - 100]

        plt.clf()
        plt.imshow(newImage)
        plt.show()

    def averageFilter(self, k):
        # Show Error message if input image did not select
        if not self.inputImg:
            QMessageBox.about(self, "Error", "Input image not found!")
            return

        # Read selected image and seperate red, green and blue
        img = mpimg.imread(self.inputImg)
        r = img[:,:,0];
        g = img[:,:,1];
        b = img[:,:,2];

        # Get image height and width
        height = r.shape[0]
        width = r.shape[1]

        # Create new image same height and width
        newImage = np.zeros((img.shape[0],img.shape[1],3), np.float64)
        # Iterate over image
        for row in np.arange(0, height):
            for col in np.arange(0, width):
                # Initialize neighbors array
                neighbors = neighborsG = neighborsB = []
                # Initialize tatol value of neighbors
                total = 0
                # Iterate over neighbors
                for m in np.arange(-k, k + 1):
                    for l in np.arange(-k, k + 1):
                        # If selected neighbors is within the borders
                        if row + m > 0 and row + m < height and col + l > 0 and col + l < width:
                            # Get neighbors value for red bands
                            a = r[row + m][col + l]
                            # Add into array
                            neighbors.append(a)

                            # Get neighbors value for green bands
                            a = g[row + m][col + l]
                            # Add into array
                            neighborsG.append(a)

                            # Get neighbors value for blue bands
                            a = b[row + m][col + l]
                            # Add into array
                            neighborsB.append(a)

                # Calculate sum of neighbors value for red band
                total = sum(neighbors)
                # Insert into new image
                newImage[...,0][row][col] = total/(k*k)

                # Calculate sum of neighbors value for green band
                total = sum(neighborsG)
                # Insert into new image
                newImage[...,1][row][col] = total/(k*k)

                # Calculate sum of neighbors value for blue band
                total = sum(neighborsB)
                # Insert into new image
                newImage[...,2][row][col] = total/(k*k)

        plt.clf()
        plt.imshow(newImage)
        plt.show()

    def medianFilter(self,k):
        # Show Error message if input image did not select
        if not self.inputImg:
            QMessageBox.about(self, "Error", "Input image not found!")
            return

        # Read selected image and seperate red, green and blue
        img = mpimg.imread(self.inputImg)
        r = img[:,:,0];
        g = img[:,:,1];
        b = img[:,:,2];

        # Get image height and width
        height = r.shape[0]
        width = r.shape[1]

        # Create new image same height and width
        newImage = np.zeros((img.shape[0],img.shape[1],3), np.float64)

        # Iterate over image
        for row in np.arange(0, height):
            for col in np.arange(0, width):
                # Initialize neighbors array
                neighbors = neighborsG = neighborsB = []
                # Iterate over neighbors
                for m in np.arange(-k, k + 1):
                    for l in np.arange(-k, k + 1):
                        # If selected neighbors is within the borders
                        if row + m > 0 and row + m < height and col + l > 0 and col + l < width:
                            # Get neighbors value for red bands
                            a = r[row + m][col + l]
                            # Add into array
                            neighbors.append(a)

                            # Get neighbors value for red green
                            a = g[row + m][col + l]
                            # Add into array
                            neighborsG.append(a)

                            # Get neighbors value for red blue
                            a = b[row + m][col + l]
                            # Add into array
                            neighborsB.append(a)

                # Find median for red band
                median = np.median(neighbors)
                # Insert into new image
                newImage[...,0][row][col] = median

                # Find median for red green
                median = np.median(neighborsG)
                # Insert into new image
                newImage[...,1][row][col] = median

                # Find median for red blue
                median = np.median(neighborsB)
                # Insert into new image
                newImage[...,2][row][col] = median

        plt.clf()
        plt.imshow(newImage)
        plt.show()

    def gaussianFilter(self, k):
        # Show Error message if input image did not select
        if not self.inputImg:
            QMessageBox.about(self, "Error", "Input image not found!")
            return

        # Read selected image and seperate red, green and blue
        img = mpimg.imread(self.inputImg)
        r = img[:,:,0];
        g = img[:,:,1];
        b = img[:,:,2];

        # Get image height and width
        height = r.shape[0]
        width = r.shape[1]

        # Intialize kernel matrix (kxk)
        kernel = [[0 for x in range(k)] for y in range(k)]
        # I selected sigma 2
        sigma = 2
        # Find half of k that is selected from user. (3x3,5x5,7x7 ...)
        center = k // 2
        # Initialize standart deviation
        x = s = 2 * sigma * sigma
        sum = 0
        # Create kernel kxk
        for i in range(-center, center+1):
            for j in range(-center, center+1):
                x = np.sqrt(i*i + j*j)
                kernel[i+center][j+center] = np.exp(-(x*x)/s) / (np.pi * s)
                sum += kernel[i+center][j+center]

        # Normalize kernel matrix
        for i in range(0,k):
            for j in range(0,k):
                kernel[i][j] = round(kernel[i][j] / sum,6)

        # Gauss filtering matrix
        gauss = np.array(kernel)

        # Half of given k. If k equal 5 we Iterate -3,+3 at the inner loop.
        half_k = k // 2

        # Create new image same height and width
        newImage = np.zeros((img.shape[0],img.shape[1],3), np.float64)
        for row in np.arange(2, height - 2):
            for col in np.arange(2, width - 2):
                # Initialize sum of neighbors of the cell
                sumR = sumG = sumB = 0
                # Iterate over neighbors of cell
                for m in np.arange(-half_k, half_k+1):
                    for l in np.arange(-half_k, half_k+1):
                        if row + m > 0 and row + m < height and col + l > 0 and col + l < width:
                            # Get neighbors for red band
                            a = r[row + m][col + l]
                            # Get gauss value
                            p = gauss[half_k + m, half_k + l]
                            # Add to sum
                            sumR = sumR + (p * a)

                            # Get neighbors for green band
                            a = g[row + m][col + l]
                            # Get gauss value
                            p = gauss[half_k + m, half_k + l]
                            # Add to sum
                            sumG = sumG + (p * a)

                            # Get neighbors for blue band
                            a = b[row + m][col + l]
                            # Get gauss value
                            p = gauss[half_k + m, half_k + l]
                            # Add to sum
                            sumB = sumB + (p * a)

                # Fill the new image from new value that is gaussian filtered value
                newImage[...,0][row][col] = sumR
                newImage[...,1][row][col] = sumG
                newImage[...,2][row][col] = sumB

        plt.clf()
        plt.imshow(newImage)
        plt.show()

if __name__ == '__main__':
    main()
