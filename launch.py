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

parser = argparse.ArgumentParser(
    description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Path to output the prediction in case of single image.')
args = parser.parse_args()
s = Launch(args)
s.caffeInference()
