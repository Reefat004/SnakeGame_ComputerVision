import cvzone
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import random

# set up video stream
videoStream = cv2.VideoCapture(0) # camera number 1
videoStream.set(3, 1280)   # set width (propID: 3) of video stream to 1280 pixels
videoStream.set(4, 720)    # set height (propID: 4) of video stream to 720 pixels

# initialize detector
# assign hand to detector if 80% or more match with object in vision
detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, appleImagePath):
        # initialize the snake
        self.gameOver = False
        self.pointsCoord = []
        self.pointsDistances = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHeadCoord = 0, 0

        imageApple = cv2.imread(appleImagePath, cv2.IMREAD_UNCHANGED)
        widthApple = int(imageApple.shape[1] * 20 / 200)   # shrink size of apple to 20%
        heightApple = int(imageApple.shape[0] * 20 / 200)

        dimensions = (widthApple, heightApple)

        self.imgApple = cv2.resize(imageApple, dimensions, interpolation=cv2.INTER_AREA)

        self.hApple, self.wApple, _ = self.imgApple.shape
        self.appleCoord = 0, 0
        self.spawnAppleRandomly()
        self.score = 0

    def spawnAppleRandomly(self):
        # randomly spawn apples within the screen
        self.appleCoord = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentSnakeHead):
        if self.gameOver:
            cv2.putText(
                imgMain,
                "Game Over",
                [50, 80],
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                thickness=5,
                color=(0, 100, 0)
            )
        else:
            # recalculate the snake
            px, py = self.previousHeadCoord
            cx, cy = currentSnakeHead
            self.pointsCoord.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)  # calculate distance in 2d space between points
            self.pointsDistances.append(distance)
            self.currentLength += distance
            self.previousHeadCoord = cx, cy

            # make snake have a length max
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.pointsDistances):
                    self.currentLength -= length
                    self.pointsDistances.pop(i)
                    self.pointsCoord.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            # eat food
            rx, ry = self.appleCoord
            if rx - self.wApple // 2 < cx < rx + self.wApple // 2 and \
                    ry - self.hApple // 2 < cy < ry + self.hApple // 2:
                self.spawnAppleRandomly()
                self.allowedLength += 50
                self.score += 1

            # draw snake
            if self.pointsCoord:
                for i, point in enumerate(self.pointsCoord):
                    if i != 0:  # head of the snake; hand so dont draw
                        cv2.line(imgMain, self.pointsCoord[i-1], self.pointsCoord[i], (0, 100, 0), 20)

            # draw food
            rx, ry = self.appleCoord
            imgMain = cvzone.overlayPNG(imgMain, self.imgApple)
            cv2.putText(
                imgMain,
                str(self.score),
                [50, 80],
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                thickness=5,
                color=(0, 100, 0)
            )

            # Check for collision
            pts = np.array(self.pointsCoord[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            if -1 <= minDist <= 1:
                self.gameOver = True
                self.pointsCoord = []
                self.pointsDistances = []
                self.currentLength = 0
                self.allowedLength = 150
                self.previousHeadCoord = 0, 0
                self.spawnAppleRandomly()
        return imgMain


# initialize the game
game = SnakeGameClass("apple.png")

while True:
    success, image = videoStream.read()   # returns 2 values; success->if video detected; image-> the video

    # read the videostream
    image = cv2.flip(image, flipCode=1)  # flipCode=1 flips the video horizontally
    # identify hands
    hands, image = detector.findHands(image, flipType=False)
    if hands:
        handCoordinates = hands[0]['lmList'][8][0:2]  # get the coordinates/position of the hands from the data returned; lmList -> landmark list (coordinates)
        image = game.update(image, handCoordinates)

    # display video
    cv2.imshow("Image", image)

    key = cv2.waitKey(2)   # search for hands after two seconds

image.release()

