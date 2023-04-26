import math

import cv2
import numpy as np
import mediapipe as mp


class ImageHandsDetection:
    def __init__(self, image, hands):
        self.__hands = hands
        self.__image = image
        self.__image_weight = 0
        self.__image_height = 0
        self.__result = None

        self.__image_weight, self.__image_height, self.__result = \
            self.__identify_hands(self.__image, self.__hands, is_flip=False)

    @staticmethod
    def __distance(x1, y1, x2, y2):
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)
        return distance

    @staticmethod
    def __vector_angle(v1, v2):
        # 定义向量
        a = np.array(v1)
        b = np.array(v2)

        # 计算向量之间的夹角
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.arccos(cos_angle)

        # 将弧度转换为角度
        angle_deg = np.degrees(angle)
        return angle_deg

    @staticmethod
    def __identify_hands(img, hands, is_flip=False):
        img_height = img.shape[0]
        img_weight = img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if is_flip:
            img = cv2.flip(img, 1)
        result = hands.process(img)
        return img_weight, img_height, result

    @staticmethod
    def __find_closest_elements(numbers):
        # 初始化最接近的两个元素及其索引
        closest_index1 = 0
        closest_index2 = 1
        # 计算初始最小差值
        min_diff = abs(numbers[0] - numbers[1])
        # 遍历列表中的每个元素
        for i in range(len(numbers) - 1):
            # 计算当前相邻元素的差值
            diff = abs(numbers[i] - numbers[i + 1])
            # 如果当前差值比之前记录的最小差值更小
            if diff < min_diff:
                min_diff = diff
                closest_index1 = i
                closest_index2 = i + 1
        # 返回最接近的两个元素及其索引
        return closest_index1, closest_index2

    def is_phone(self, show_hand=False):
        result = False
        if self.__result.multi_hand_landmarks:
            for handLms in self.__result.multi_hand_landmarks:
                if show_hand:
                    mp.solutions.drawing_utils.draw_landmarks(self.__image, handLms,
                                                              mp.solutions.hands.HAND_CONNECTIONS)
                hand_mark = handLms.landmark
                dis5_6 = self.__distance(hand_mark[5].x, hand_mark[5].y, hand_mark[6].x, hand_mark[6].y)
                dis9_10 = self.__distance(hand_mark[9].x, hand_mark[9].y, hand_mark[10].x, hand_mark[10].y)
                dis13_14 = self.__distance(hand_mark[13].x, hand_mark[13].y, hand_mark[14].x, hand_mark[14].y)
                dis17_18 = self.__distance(hand_mark[17].x, hand_mark[17].y, hand_mark[18].x, hand_mark[18].y)
                dis7_8 = self.__distance(hand_mark[7].x, hand_mark[7].y, hand_mark[8].x, hand_mark[8].y)
                dis11_12 = self.__distance(hand_mark[11].x, hand_mark[11].y, hand_mark[12].x, hand_mark[12].y)
                dis15_16 = self.__distance(hand_mark[15].x, hand_mark[15].y, hand_mark[16].x, hand_mark[16].y)
                dis19_20 = self.__distance(hand_mark[19].x, hand_mark[19].y, hand_mark[20].x, hand_mark[20].y)
                if dis7_8 / dis5_6 < 0.3 and dis11_12 / dis9_10 < 0.3 \
                        and dis15_16 / dis13_14 < 0.3 and dis19_20 / dis17_18 < 0.3 \
                        and abs(hand_mark[5].x - hand_mark[6].x) / dis5_6 > 0.7 \
                        and abs(hand_mark[9].x - hand_mark[10].x) / dis9_10 > 0.7 \
                        and abs(hand_mark[13].x - hand_mark[14].x) / dis13_14 > 0.7 \
                        and abs(hand_mark[17].x - hand_mark[18].x) / dis17_18 > 0.7:
                    result = True
        return result

    def is_bowl(self, show_hand=False):
        hand_list = []
        result = False
        if self.__result.multi_hand_landmarks:
            for handLms in self.__result.multi_hand_landmarks:
                if show_hand:
                    mp.solutions.drawing_utils.draw_landmarks(self.__image, handLms,
                                                              mp.solutions.hands.HAND_CONNECTIONS)
                hand_mark = handLms.landmark
                dis5_17 = self.__distance(hand_mark[5].x, hand_mark[5].y, hand_mark[17].x, hand_mark[17].y)
                hand_list.append(dis5_17)
            if len(hand_list) >= 2:
                index1, index2 = self.__find_closest_elements(hand_list)
                handLms1 = self.__result.multi_hand_landmarks[index1]
                handLms2 = self.__result.multi_hand_landmarks[index2]
                points1 = handLms1.landmark
                points2 = handLms2.landmark
                dis = (self.__distance(points1[5].x, points1[5].y, points1[17].x, points1[17].y)
                       + self.__distance(points2[5].x, points2[5].y, points2[17].x, points2[17].y)) / 2
                dis17_17 = self.__distance(points1[17].x, points1[17].y, points2[17].x, points2[17].y)
                dis18_18 = self.__distance(points1[18].x, points1[18].y, points2[18].x, points2[18].y)
                dis20_20 = self.__distance(points1[20].x, points1[20].y, points2[20].x, points2[20].y)
                dis5_5 = self.__distance(points1[5].x, points1[5].y, points2[5].x, points2[5].y)
                dis8_8 = self.__distance(points1[8].x, points1[8].y, points2[8].x, points2[8].y)
                if dis17_17 / dis < 0.7 and dis18_18 / dis < 0.6 and dis20_20 / dis < 0.6 and \
                        dis5_5 / dis > 1.9 and dis8_8 / dis < 1.8:
                    if points1[12].y > points1[9].y and points1[0].y > points1[9].y and points2[12].y > points2[9].y and \
                            points2[0].y > points2[9].y:
                        result = True
        return result

    def is_open_hand(self):
        result = False
        if self.__result.multi_hand_landmarks:
            for index, handLms in enumerate(self.__result.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(self.__image, handLms,
                                                          mp.solutions.hands.HAND_CONNECTIONS)
                hand_mark = handLms.landmark
                palm_len = self.__distance(hand_mark[9].x, hand_mark[9].y, hand_mark[0].x, hand_mark[0].y)
                dis12_0 = self.__distance(hand_mark[12].x, hand_mark[12].y, hand_mark[9].x, hand_mark[9].y)
                dis8_5 = self.__distance(hand_mark[8].x, hand_mark[8].y, hand_mark[5].x, hand_mark[5].y)
                dis16_13 = self.__distance(hand_mark[16].x, hand_mark[16].y, hand_mark[13].x, hand_mark[13].y)
                dis20_17 = self.__distance(hand_mark[20].x, hand_mark[20].y, hand_mark[17].x, hand_mark[17].y)
                dis5_17 = self.__distance(hand_mark[5].x, hand_mark[5].y, hand_mark[17].x, hand_mark[17].y)
                v1 = (hand_mark[4].x - hand_mark[3].x, hand_mark[4].y - hand_mark[3].y)
                v2 = (hand_mark[2].x - hand_mark[3].x, hand_mark[2].y - hand_mark[3].y)
                angle432 = self.__vector_angle(v1, v2)
                if angle432 > 160 and dis12_0 / palm_len > 0.85 and dis8_5 / palm_len > 0.8 \
                        and dis16_13 / palm_len > 0.8 and dis20_17 / palm_len > 0.75 and dis5_17 / palm_len > 0.6:
                    result = True
        return result

    def is_grab_hand(self, show_hand=False):
        result = False
        if self.__result.multi_hand_landmarks:
            for index, handLms in enumerate(self.__result.multi_hand_landmarks):
                hand_mark = handLms.landmark
                palm_len = self.__distance(hand_mark[9].x, hand_mark[9].y, hand_mark[0].x, hand_mark[0].y)
                dis12_0 = self.__distance(hand_mark[12].x, hand_mark[12].y, hand_mark[9].x, hand_mark[9].y)
                dis8_5 = self.__distance(hand_mark[8].x, hand_mark[8].y, hand_mark[5].x, hand_mark[5].y)
                dis16_13 = self.__distance(hand_mark[16].x, hand_mark[16].y, hand_mark[13].x, hand_mark[13].y)
                dis20_17 = self.__distance(hand_mark[20].x, hand_mark[20].y, hand_mark[17].x, hand_mark[17].y)
                dis5_17 = self.__distance(hand_mark[5].x, hand_mark[5].y, hand_mark[17].x, hand_mark[17].y)
                v1 = (hand_mark[4].x - hand_mark[3].x, hand_mark[4].y - hand_mark[3].y)
                v2 = (hand_mark[2].x - hand_mark[3].x, hand_mark[2].y - hand_mark[3].y)
                angle432 = self.__vector_angle(v1, v2)
                if hand_mark[9].z < hand_mark[10].z and angle432 < 150 and dis12_0 / palm_len < 0.55 \
                        and dis8_5 / palm_len < 0.55 and dis16_13 / palm_len < 0.55 \
                        and dis20_17 / palm_len < 0.55 and dis5_17 / palm_len > 0.6:
                    result = True
        return result

    def is_ok(self):
        result = False
        if self.__result.multi_hand_landmarks:
            for handLms in self.__result.multi_hand_landmarks:
                hand_mark = handLms.landmark

                v1 = (hand_mark[8].x - hand_mark[5].x, hand_mark[8].y - hand_mark[5].y)
                v2 = (hand_mark[0].x - hand_mark[5].x, hand_mark[0].y - hand_mark[5].y)
                angle850 = self.__vector_angle(v1, v2)

                v3 = (hand_mark[12].x - hand_mark[9].x, hand_mark[12].y - hand_mark[9].y)
                v4 = (hand_mark[0].x - hand_mark[9].x, hand_mark[0].y - hand_mark[9].y)
                angle1290 = self.__vector_angle(v3, v4)

                v5 = (hand_mark[4].x - hand_mark[0].x, hand_mark[4].y - hand_mark[0].y)
                v6 = (hand_mark[8].x - hand_mark[0].x, hand_mark[8].y - hand_mark[0].y)
                angle408 = self.__vector_angle(v5, v6)

                v7 = (hand_mark[16].x - hand_mark[13].x, hand_mark[16].y - hand_mark[13].y)
                v8 = (hand_mark[0].x - hand_mark[13].x, hand_mark[0].y - hand_mark[13].y)
                angle16130 = self.__vector_angle(v7, v8)

                v9 = (hand_mark[20].x - hand_mark[17].x, hand_mark[20].y - hand_mark[17].y)
                v10 = (hand_mark[0].x - hand_mark[17].x, hand_mark[0].y - hand_mark[17].y)
                angle20170 = self.__vector_angle(v9, v10)
                if angle850 <= 130 and angle1290 > 150 and angle20170 > 150 and angle16130 > 150 and angle408 < 25:
                    result = True
        return result

    def index_finger_pointing_at(self, show_hand=False):
        points = []
        if self.__result.multi_hand_landmarks:
            for handLms in self.__result.multi_hand_landmarks:
                if show_hand:
                    mp.solutions.drawing_utils.draw_landmarks(self.__image, handLms,
                                                              mp.solutions.hands.HAND_CONNECTIONS)
                hand_mark = handLms.landmark
                palm_len = self.__distance(hand_mark[5].x, hand_mark[5].y, hand_mark[0].x, hand_mark[0].y)
                v1 = (hand_mark[8].x - hand_mark[7].x, hand_mark[8].y - hand_mark[7].y)
                v2 = (hand_mark[6].x - hand_mark[7].x, hand_mark[6].y - hand_mark[7].y)
                angle6_7_8 = self.__vector_angle(v1, v2)
                v3 = (hand_mark[4].x - hand_mark[1].x, hand_mark[4].y - hand_mark[1].y)
                v4 = (hand_mark[0].x - hand_mark[1].x, hand_mark[0].y - hand_mark[1].y)
                angle0_1_4 = self.__vector_angle(v3, v4)
                if angle6_7_8 > 160 and angle0_1_4 < 150 and hand_mark[9].z < hand_mark[10].z and \
                        self.__distance(hand_mark[5].x, hand_mark[5].y, hand_mark[8].x,
                                        hand_mark[8].y) / palm_len > 0.6 and \
                        self.__distance(hand_mark[12].x, hand_mark[12].y, hand_mark[0].x,
                                        hand_mark[0].y) / palm_len < 1 and \
                        self.__distance(hand_mark[16].x, hand_mark[16].y, hand_mark[0].x,
                                        hand_mark[0].y) / palm_len < 1 and \
                        self.__distance(hand_mark[20].x, hand_mark[20].y, hand_mark[0].x,
                                        hand_mark[0].y) / palm_len < 1:
                    points.append(hand_mark)
        if len(points) == 1:
            return points[0][8].x, points[0][8].y
        if len(points) > 1:
            return points[0][8].x, points[0][8].y, points[1][8].x, points[1][8].y
