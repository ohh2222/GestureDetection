from enum import Enum
import time
from ImageHandsDetection import *


class OpenGrabStatus(Enum):
    status0 = 0
    status1 = 1
    status2 = 2


class VideoHandsDetection:

    def __init__(self, hands):
        self.__hands = hands
        self.__open_grab_status = OpenGrabStatus.status1
        self.__count1 = 0
        self.__cTime1 = time.time()
        self.__pTime1 = time.time()

        self.__points = []
        self.__count2 = 0
        self.__old_point = (0, 0)
        self.__cTime2 = time.time()
        self.__pTime2 = time.time()

    @staticmethod
    def __distance(x1, y1, x2, y2):
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)
        return distance

    def is_open_grab_detection(self, image):
        result = False
        self.__cTime1 = time.time()
        # pTime上次循环结束的时间
        fps = int(1 / (self.__cTime1 - self.__pTime1))
        self.__pTime1 = self.__cTime1
        hand_detection = ImageHandsDetection(image, self.__hands)
        if hand_detection.is_grab_hand():
            self.__open_grab_status = OpenGrabStatus.status0
        if self.__open_grab_status == OpenGrabStatus.status0:
            self.__count1 += 1
        if self.__count1 >= 3 * fps:
            self.__open_grab_status = OpenGrabStatus.status1
            self.__count1 = 0
        if self.__open_grab_status == OpenGrabStatus.status0 and hand_detection.is_open_hand():
            self.__open_grab_status = OpenGrabStatus.status1
            self.__count1 = 0
            result = True
        return result

    # 食指按压状态停留点
    def index_finger_stay(self, image):
        self.__cTime2 = time.time()
        # pTime上次循环结束的时间
        fps = int(1 / (self.__cTime2 - self.__pTime2))
        self.__pTime2 = self.__cTime2
        hand_detection = ImageHandsDetection(image, self.__hands)
        point = hand_detection.index_finger_pointing_at()
        if point:
            if len(point) == 2:
                if self.__distance(self.__old_point[0], self.__old_point[1], point[0], point[1]) > 0.005:
                    self.__points.clear()
                self.__old_point = point
                self.__count2 = 0
                self.__points.append(point)
            else:
                self.__count2 += 1
        else:
            self.__count2 += 1
        # 非按压手势和一只手以上按压超过0.5s清零重新计算
        if self.__count2 > 0.5 * fps:
            self.__points.clear()
            self.__count2 = 0
        # 停留1.5s
        if len(self.__points) > 1.5 * fps:
            data = np.array(self.__points)
            self.__points.clear()
            self.__count2 = 0
            # 计算方差
            xy_variance = np.var(data[:, 0]) + np.var(data[:, 1])
            x_mean = int(np.mean(data[:, 0])*image.shape[1])
            y_mean = int(np.mean(data[:, 1])*image.shape[0])
            if xy_variance * 100 <= 0.05:
                return x_mean, y_mean
