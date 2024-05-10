import rclpy
from rclpy.node import Node
import math
from custom_interfaces.msg import Vision, VisionVector, VisionVector1, VisionVector2
import sys
sys.path.insert(0, './src/vision_yolov7/vision_yolov7')
import numpy as np
from numpy import random
import cv2
import torch
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from models.experimental import attempt_load
from .ClassConfig import *

THRESHOLD = 0.45

class LandmarkDetection(Node):
    def __init__(self, config):
        super().__init__('landmark_detection')
        self.config = config
        self.publisher_centerlandmark = self.create_publisher(VisionVector, '/centerlandmark_position', 10)
        self.publisher_penaltilandmark = self.create_publisher(VisionVector1, '/penaltilandmark_position', 10)
        self.publisher_goalpostlandmark = self.create_publisher(VisionVector2, '/goalpostlandmark_position', 10)
        self.weights = 'src/vision_yolov7/vision_yolov7/peso_tiny/best_localization.pt'
        self.detect_landmarks()
        
    def detect_landmarks(self):
        set_logging()
        device = select_device('cpu')
        msg_centerlandmark=VisionVector()
        msg_penaltilandmark=VisionVector1()
        msg_goalpostlandmark=VisionVector2()
        #Load modelo com o peso dos landmarks
        model = attempt_load(self.weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(640, s=stride)
        #Nomes das classes
        names = model.names
        #Cores
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        #Camera
        dataset = LoadStreams('/dev/video0', img_size=imgsz, stride=stride)
        
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) #Executar uma vez
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device).float() / 255.0 #uint8 to fp32  /  0 - 255 para 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img)[0]
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=True)
            im0, frame = im0s[0].copy(), dataset.count

            msg_centerlandmark.detected = msg_penaltilandmark.detected = msg_goalpostlandmark.detected = False
            msg_centerlandmark.left = msg_penaltilandmark.left = msg_goalpostlandmark.left = False
            msg_centerlandmark.center_left = msg_penaltilandmark.center_left = msg_goalpostlandmark.center_left = False
            msg_centerlandmark.center_right = msg_penaltilandmark.center_right = msg_goalpostlandmark.center_right = False
            msg_centerlandmark.right = msg_penaltilandmark.right = msg_goalpostlandmark.right = False
            msg_centerlandmark.med = msg_penaltilandmark.med = msg_goalpostlandmark.med = False
            msg_centerlandmark.far = msg_penaltilandmark.far = msg_goalpostlandmark.far = False
            msg_centerlandmark.close = msg_penaltilandmark.close = msg_goalpostlandmark.close = False
            
            if pred[0] is not None:
                for *xyxy, conf, cls in reversed(pred[0]):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    if conf > THRESHOLD:
                        c1_center = (xyxy[0] + xyxy[2]) / 2
                        c2_center = (xyxy[1] + xyxy[3]) / 2
                        if names[int(cls)] in ("center", "penalti", "goalpost"):
                            msg = None
                            if names[int(cls)] == "center":
                                msg = msg_centerlandmark
                            elif names[int(cls)] == "penalti":
                                msg = msg_penaltilandmark
                            else:
                                msg = msg_goalpostlandmark
                            if msg is not None:
                                msg.detected = True
                                x_pos = "left" if int(c1_center) <= self.config.x_left else \
                                        "center_left" if int(c1_center) < self.config.x_center else \
                                        "center_right" if int(c1_center) > self.config.x_center and int(c1_center) < self.config.x_right else \
                                        "right"
                                y_pos = "far" if int(c2_center) <= self.config.y_longe else \
                                        "close" if int(c2_center) >= self.config.y_chute else \
                                        "med"
                                setattr(msg, x_pos, True)
                                setattr(msg, y_pos, True)
                                getattr(self, f"publisher_{names[int(cls)]}landmark").publish(msg)
                                print(f"{names[int(cls)]} detectado: {msg.detected}, {x_pos}, {y_pos}")
                                #Calculando a distância se estiver em center_right/med ou center_left/med (landmarks centralizados na câmera)
                                #if (x_pos == "center_right" or x_pos == "center_left") and y_pos == "med":
                                #    angle = motor_angle(angulo_do_motor)
                                #    distance = calculate_distance(angle)
                                #    print(f"Distância entre robô e landmark {names[int(cls)]}: {distance}")
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
            cv2.imshow('Landmark Detection', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#def motor_angle(x):
#    return (((x-1024)*90)/1024)

#def calculate_distance(angle):
#    camera_height = 0.5  #Altura do motor do pescoço até a câmera em metros
#    robot_height = 2  #Altura do robô (até o pescoço)
#    angle_rad = math.radians(angle) #Ângulo em radianos

#    y = camera_height * math.sin(angle_rad)
#    x = camera_height * math.cos(angle_rad)

#    total_height = high + y
#    distance = math.tan(angle_rad) * total_height + x
#    return distance

def main(args=None):
    rclpy.init(args=args)
    config = classConfig()
    landmark_detection = LandmarkDetection(config)
    rclpy.spin(landmark_detection)
    landmark_detection.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()