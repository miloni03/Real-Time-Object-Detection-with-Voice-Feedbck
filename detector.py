import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import keyboard
import pyttsx3
import os
import math
import pyttsx3
import pickle
engine = pyttsx3.init()

class detector:
    def __init__(self, videopath, configpath, modelpath, classespath) -> None:
        self.videopath = videopath
        self.configpath = configpath
        self.modelpath = modelpath
        self.classespath = classespath

        self.net = cv2.dnn_DetectionModel(self.modelpath, self.configpath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean(127.5)
        self.net.setInputSwapRB(True)

        self.readClasses()

        self.engine = pyttsx3.init()
        self.spoken_objects = set()

        self.focal_length = 2000  # Change this value according to your webcam

    def readClasses(self):
        with open(self.classespath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, "_Backgroound_")
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def preprocess(self, image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)

        brightened = cv2.add(sharpened, 50)

        gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)

        return edges


    def onVideo(self):
        cap = cv2.VideoCapture(self.videopath)
        if (cap.isOpened() == False):
            print("error opening the file.....")
            return
        success, image = cap.read()
        
        while success:
            classLabelIds, confidences, bboxs = self.net.detect(image, confThreshold=0.5)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            print("hello")
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)
            print("hi")


            for i in range(len(bboxIdx)):
                bbox = bboxs[np.squeeze(bboxIdx[i])]
                classConfidence = confidences[np.squeeze(bboxIdx[i])]
                classLabelId = classLabelIds[np.squeeze(bboxIdx[i])]
                classLabel = self.classesList[classLabelId]
                classColor = [int(c) for c in self.colorList[classLabelId]]

                displayText = "{}".format(classLabel)
                print("yo")

                x, y, w, h = bbox
                cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, 2)
                cv2.imshow("result",image)
                
                mid_x = x + w // 2
                mid_y = y + h // 2
                distance = self.get_distance(mid_x, mid_y, image.shape[1] // 2, image.shape[0] // 2)
                distance = round(distance / self.focal_length, 2)
                

                # Speak the distance if the object is not already spoken for 5 seconds
                if classLabel not in self.spoken_objects:
                    text = "Found {} at {:.2f} meters".format(classLabel, distance)
                    engine.say(text)
                    engine.runAndWait()
                    self.spoken_objects.add(classLabel)

            
            if keyboard.is_pressed("a"):
                break

            # Clearing the spoken_objects set every 5 seconds
            if int(time.time()) % 5 == 0:
                self.spoken_objects.clear()


        cap.release()
        cv2.destroyAllWindows()

    
def main():
    videopath = 0 #'Vatsal DS entry.mp4'
    configpath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelpath = os.path.join("model_data", "frozen_inference_graph.pb")
    classespath = os.path.join("model_data", "coco.names")
    p = detector(videopath, configpath, modelpath, classespath)
    p.onVideo()


if __name__ == "__main__":
    main()


