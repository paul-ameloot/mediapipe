#!/usr/bin/env python3

import rclpy, numpy as np
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from mediapipe_gesture_recognition.msg import Hand, Pose

class Pointer(Node):

    def __init__(self, node_name='pointer_subscriber'):

        # Inizializzazione del nodo
        super().__init__(node_name)

        # Inizializzazione dei topic
        self.right_subscriber = self.create_subscription(Hand, '/mediapipe_gesture_recognition/right_hand', self.right_hand_callback, 10)
        self.left_subscriber = self.create_subscription(Hand, '/mediapipe_gesture_recognition/left_hand', self.left_hand_callback, 10)
        self.pose_subscriber = self.create_subscription(Pose, '/mediapipe_gesture_recognition/pose', self.pose_callback, 10)

        self.area_pub = self.create_publisher(Int32MultiArray, 'area', 1)

        # TODO: publish all the items and areas on fusion node

        self.right_msg = None
        self.left_msg  = None
        self.pose_msg  = None
        self.Area_threshold = 1.8
        self.Item_threshold = 0.2
        self.z = -2

    def find_line_equation(self, p1, p2):

        # Trova l'equazione della retta che passa per i punti p1 e p2
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        m = dz / ((dx ** 2 + dy ** 2 + dz ** 2) ** 0.5)
        q = z1 - m * ((dx ** 2 + dy ** 2) ** 0.5)

        return m, q

    def find_point_on_line(self, p1, p2):

        #Trova il punto sulla retta che passa per i punti p1 e p2 con la coordinata z data

        m, q = self.find_line_equation(p1, p2)

        x1, y1, z1 = p1

        x = ((self.z - q) / m) * ((p2[0] - p1[0]) / ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**0.5) + x1
        y = ((self.z - q) / m) * ((p2[1] - p1[1]) / ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**0.5) + y1

        p3 = [x, y, self.z]

        return p3

    # Callback Functions
    def RightHandCallback(self, data): self.right_msg: Hand() = data
    def LeftHandCallback(self, data):  self.left_msg:  Hand() = data
    def PoseCallback(self, data):      self.pose_msg:  Pose() = data

    def create3DLine_Right(self):

        # print(f"{'Nr':<3} | {self.pose_msg.keypoints[16].keypoint_name}")
        # print(f"{'X':<3} | {self.pose_msg.keypoints[16].x:.2f}")
        # print(f"{'Y':<3} | {self.pose_msg.keypoints[16].y:.2f}")
        # print(f"{'Z':<3} | {self.pose_msg.keypoints[16].depth:.2f} \n")

        # print(f"{'Nr':<3} | {self.pose_msg.keypoints[14].keypoint_name}")
        # print(f"{'X':<3} | {self.pose_msg.keypoints[14].x:.2f}")
        # print(f"{'Y':<3} | {self.pose_msg.keypoints[14].y:.2f}")
        # print(f"{'Z':<3} | {self.pose_msg.keypoints[14].depth:.2f} \n")

        # Definizione delle coordinate
        x1 = float(self.pose_msg.keypoints[15].x)
        y1 = float(self.pose_msg.keypoints[15].y)
        z1 = float(self.pose_msg.keypoints[15].depth)
        x2 = float(self.pose_msg.keypoints[13].x)
        y2 = float(self.pose_msg.keypoints[13].y)
        z2 = float(self.pose_msg.keypoints[13].depth)

        # Definizione dei punti
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])

        p3_R = self.find_point_on_line(p1, p2)

        Area1 = np.array([-1, -0.5, -2])
        Area2 = np.array([3, 0.5, -2])

        distancefromArea1 = np.linalg.norm(np.cross(p2-p1, p1 - Area1)) / np.linalg.norm(p2 - p1)
        distancefromArea2 = np.linalg.norm(np.cross(p2-p1, p2 - Area2)) / np.linalg.norm(p2-p1)

        print("\n\n\n\n\n")
        print("Distance Area 1 destra: ", distancefromArea1)
        print("Distance Area 2 destra : ", distancefromArea2)
        print("P3_R coordinates: ", p3_R)
        print("\n\n\n\n\n")

        areamsg = Int32MultiArray()

        #  # Stampa se il punto appartiene all'intorno della retta
        # if distancefromArea1 <= self.Area_threshold:
        #     print("Stai indicando l'Area 2 con la destra")
        #     areamsg.data = [2]
        #     self.area_pub.publish(areamsg)
        #     return areamsg.data

        if distancefromArea2 <= 2.2:
            print("Stai indicando l'Area 1 con la destra")
            areamsg.data = [1]
            self.area_pub.publish(areamsg)
            return areamsg.data

    def create3DLine_Left(self):

        if not self.pose_msg == None:

            # print(f"{'Nr':<3} | {self.pose_msg.keypoints[15].keypoint_name}")
            # print(f"{'X':<3} | {self.pose_msg.keypoints[15].x:.2f}")
            # print(f"{'Y':<3} | {self.pose_msg.keypoints[15].y:.2f}")
            # print(f"{'Z':<3} | {self.pose_msg.keypoints[15].depth:.2f} \n")

            # print(f"{'Nr':<3} | {self.pose_msg.keypoints[13].keypoint_name}")
            # print(f"{'X':<3} | {self.pose_msg.keypoints[13].x:.2f}")
            # print(f"{'Y':<3} | {self.pose_msg.keypoints[13].y:.2f}")
            # print(f"{'Z':<3} | {self.pose_msg.keypoints[13].depth:.2f} \n")

            # Definizione delle coordinate
            x1 = float(self.pose_msg.keypoints[16].x)
            y1 = float(self.pose_msg.keypoints[16].y)
            z1 = float(self.pose_msg.keypoints[16].depth)

            x2 = float(self.pose_msg.keypoints[14].x)
            y2 = float(self.pose_msg.keypoints[14].y)
            z2 = float(self.pose_msg.keypoints[14].depth)

            # Definizione dei punti
            p1 = np.array([x1, y1, z1])
            p2 = np.array([x2, y2, z2])

            p3_L = self.find_point_on_line(p1, p2)

            #Definiamo le coordinate di tre aree
            Area1 = np.array([-1, -0.5, -2])
            Area2 = np.array([3, 0.5, -2])

            # Calcolo della distanza tra l'area puntata e le aree
            distancefromArea1 = np.linalg.norm(np.cross(p2-p1, p1 - Area1)) / np.linalg.norm(p2 - p1)
            distancefromArea2 = np.linalg.norm(np.cross(p2-p1, p2 - Area2)) / np.linalg.norm(p2-p1)

            print("\n\n")
            print("Distanza sinistra Area 1: ", distancefromArea1)
            print("Distanza sinistra Area 2: ", distancefromArea2)
            print("P3_L coordinates: ", p3_L)
            print("\n\n")

            # itemmsg = Float32MultiArray()
            areamsg = Int32MultiArray()

            # Stampa se il punto appartiene all'intorno della retta
            if distancefromArea1 <= self.Area_threshold:
                print("Stai indicando l'Area 2 con la sinistra")
                areamsg.data = [2]
                self.area_pub.publish(areamsg)
                return areamsg.data

            # if distancefromArea2 <= self.Area_threshold:
            #     print("Stai indicando l'Area 1 con la sinistra")
            #     areamsg.data = [1]
            #     self.area_pub.publish(areamsg)
            #     return areamsg.data
            print("\n\n")

if __name__ == '__main__':

    P = Pointer()

    # mode = input("Quale configurazione vuoi utilizzare? Destra (R), Sinistra (L) o Entrambe (B)?")
    mode = 'B'

    while not rclpy.ok():

        rclpy.sleep(0.25)

        if mode == "R" or mode == 'r' or mode == "right" or mode == "destra" or mode == 'D' or mode == 'd':

            P.create3DLine_Right()

        elif mode == "L" or mode == 'l' or mode == "left" or mode == "sinistra" or mode == 'S' or mode == 's':

            P.create3DLine_Left()

        elif mode == "B" or mode == 'b' or mode == "both" or mode == "entrambe" or mode == 'E' or mode == 'e':

            Area_sinistra = P.create3DLine_Left()
            Area_destra = P.create3DLine_Right()

            print(Area_sinistra, Area_destra)
