import time
import math
import argparse
import cv2
import sys
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Launch:
    def __init__(self, args):
        self.args = args
        self.ageList = ['(0-3)', '(4-7)', '(8-15)', '(16-20)',
                        '(21-25)', '(26-30)', '(31-43)',
                         '(48-53)',
                         '(60-80)']
        self.ages = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(21-24)", "(25-32)",
                     "(33-37)", "(38-43)", "(44-47)", "(48-53)", "(54-59)", "(60-100)"]
        self.genders = ["Male", "Female"]
        faceProto = "models/face/opencv_face_detector.pbtxt"
        faceModel = "models/face/opencv_face_detector_uint8.pb"
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageProto = "models/age/age_deploy.prototxt"
        ageModel = "models/age/age_net.caffemodel"
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderProto = "models/gender/gender_deploy.prototxt"
        genderModel = "models/gender/gender_net.caffemodel"
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    @staticmethod
    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                     104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                              (0, 255, 0), int(round(frameHeight / 150)), 8)

        return frameOpencvDnn, bboxes
    
    
    def caffeInference(self):
        cap = cv2.VideoCapture(self.args.input if self.args.input else 0)
        padding = 20
        while cv2.waitKey(1) < 0:
            t = time.time()
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv2.waitKey()
                break
            frameFace, bboxes = self.getFaceBox(self.faceNet, frame)
            if not bboxes:
                print("No face Detected, Checking next frame")
                cv2.putText(frameFace, "NO FACE DETECTED!", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow("Age Gender Demo", frameFace)
            else:
                for bbox in bboxes:
                    face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                                 max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
                    blob = cv2.dnn.blobFromImage(
                        face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                    self.genderNet.setInput(
                        blob)
                    genderPreds = self.genderNet.forward()
                    gender = self.genders[genderPreds[0].argmax()]
                    print("Gender : {}, conf = {:.3f}".format(
                        gender, genderPreds[0].max()))
                    self.ageNet.setInput(blob)
                    agePreds = self.ageNet.forward()
                    age = self.ageList[agePreds[0].argmax()]
                    print("Age : {}, conf = {:.3f}".format(
                        age, agePreds[0].max()))
                    label = "{},{}".format(gender, age)
                    cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),
                                2,
                                cv2.LINE_AA)
                    if self.args.output != "":
                        filename = "output/predictions/" + str(args.output)
                        cv2.imwrite(filename, frameFace)
                    cv2.imshow("Age Gender Demo", frameFace)
            print("time : {:.3f}".format(time.time() - t))
        cv2.destroyAllWindows()

parser = argparse.ArgumentParser(
    description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Path to output the prediction in case of single image.')
args = parser.parse_args()
s = Launch(args)
s.caffeInference()
