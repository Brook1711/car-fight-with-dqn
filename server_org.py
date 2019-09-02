import numpy as np
import cv2
import socket
from collections import deque
import time
import math
import os
import ast,json
import string
from dqn import DQNAgent
#209 88 74
#200-210 80-90 70-80
#10-20 40-50 90-100
#5-10 50-70 90-100
the_ip = '192.168.43.203'
server_addr = (the_ip, 8888)
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_tcp.connect(server_addr)

start_time = time.time()
#socket_tcp.send(str(start_time).encode("utf-8"))
count =0 



class VideoStreamingTest(object):
    def __init__(self, host, port):     
        self.state_size = 3
        self.action_size = 7
        self.done = False
        self.batch_size = 32
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.state_now =np.reshape([ 0.10606659, -0.52737298,  0.47917915], [1, self.state_size])
        self.state_last = np.reshape([ 0.10606659, -0.52737298,  0.47917915], [1, self.state_size])
        self.action_for_next = 0
        self.action_for_now = 0
        self.reward = 0
        self.forward = "T394"
        self.left = "S450"
        self.right = "S270"
        self.backward = "T330"
        self.stop = "T370"
        self.middle = "S360"
        #dqn parameters
        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile("rb")
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.temp_result = None
        self.finnal_result = None
        self.RANGE = 350
        self.WIDTH = 720
        self.time_now = 0
        self.count =0
        self.streaming()

    def dqn_loop(self):
        if self.finnal_result['me']['r'] > 1:
            self.done = True
        else:
            self.done = False
        if True:
            self.prepare_state()#更新前一次状态，并获取这一次状态
            self.prepare_action()#更新前一次动作，并获取本次操作

            if self.count == 1:
                self.prepare_reward()#获取上一次活动的奖励
            else:
                self.count+=1
            self.act_move()#更新小车运动状态
            if self.count == 1:
                self.remember_step()#收集本次数据
            if len(self.agent.memory) > self.batch_size:
                self.agent.replay(self.batch_size)

    def prepare_state(self):
        self.state_last = self.state_now
        state_now_ = [self.finnal_result['me']['alpha_big'], \
        self.finnal_result['me']['alpha_small'], \
        self.finnal_result['me']['r']]
        self.state_now = np.reshape(state_now_, [1, self.state_size])
        #self.state_now = state_now_
        
    def prepare_action(self):
        self.action_for_now = self.action_for_next
        self.action_for_next = self.agent.act(self.state_now)
    
    def prepare_reward(self):#运行条件：state_last非空
        if self.done:
            self.reward = -10
        else:
            self.reward = (self.state_last[0][2] - self.state_now[0][2])*100
            #self.reward = (self.state_last[2] - self.state_now[2])*100
    def remember_step(self):
        self.agent.remember(self.state_last, self.action_for_now, self.reward, self.state_now, self.done)

    def act_move(self):
        if self.done:
            self.action_for_next = 0

        if self.action_for_next == 0:#停止
            str_S = self.middle
            str_T = self.stop
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)


        elif self.action_for_next == 1:#前进
            str_S = self.middle
            str_T = self.forward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)

        elif self.action_for_next == 2:#左转前进
            str_S = self.left
            str_T = self.forward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)

        elif self.action_for_next == 3:#右转前进
            str_S = self.right
            str_T = self.forward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)

        elif self.action_for_next == 4:#后退
            str_S = self.middle
            str_T = self.backward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)
            str_S = self.middle
            str_T = self.stop
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)
            str_S = self.middle
            str_T = self.backward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)

        elif self.action_for_next == 5:#左转后退
            str_S = self.left
            str_T = self.backward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)
            str_S = self.left
            str_T = self.stop
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            str_S = self.left
            str_T = self.backward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")

        elif self.action_for_next == 6:#右转后退
            str_S = self.right
            str_T = self.backward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)
            str_S = self.right
            str_T = self.stop
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T) 
            str_S = self.right
            str_T = self.backward
            str_S = str_S.encode("utf-8")
            str_T = str_T.encode("utf-8")
            socket_tcp.send(str_S)
            socket_tcp.send(str_T)           

    def get_one_car(self, x1, y1, x2, y2):
        x0 = (x1 + x2)/2
        y0 = (y1 + y2)/2
        detx = x1 - x2
        dety = y1 - y2
        temp_x0 = x0 - self.WIDTH/2
        temp_y0 = y0 - self.WIDTH/2
        if detx > 0:
            alpha_small = math.atan(dety/detx)
        elif detx < 0:
            alpha_small = math.atan(dety/detx) + math.pi
        else:
            if dety > 0:
                alpha_small = math.pi/2
            else:
                alpha_small = 0-math.pi/2
            

        

        if temp_x0 > 0:
            alpha_big = math.atan(temp_y0/temp_x0)
        elif temp_x0 < 0:
            alpha_big = math.atan(temp_y0/temp_x0) + math.pi
        else:
            if temp_y0 > 0:
                alpha_big = math.pi/2
            else:
                alpha_big = 0-math.pi/2

        alpha_small = alpha_small/math.pi - 0.5
        alpha_big = alpha_big/math.pi - 0.5
        r = math.sqrt(temp_x0**2 + temp_y0**2)/self.RANGE
        return {"alpha_big" : alpha_big,
                "alpha_small":alpha_small,
                "r" : r,
                "x0": x0,
                "y0": y0}


    def get_finnal_result(self):
        red_x = self.temp_result["red"]["x"]
        red_y = self.temp_result["red"]["y"]
        green_x = self.temp_result["green"]["x"]
        green_y = self.temp_result["green"]["y"]
        blue_x = self.temp_result["blue"]["x"]
        blue_y = self.temp_result["blue"]["y"]
        yellow_x = self.temp_result["yellow"]["x"]
        yellow_y = self.temp_result["yellow"]["y"]
        finnal_temp = {}
        me_temp = self.get_one_car(red_x, red_y, green_x, green_y)
        enemy_temp = self.get_one_car(blue_x, blue_y, yellow_x, yellow_y)
        finnal_temp["me"] = me_temp
        finnal_temp["enemy"] = enemy_temp
        self.finnal_result = finnal_temp

    def draw(self, frame, lowerRGB, upperRGB, word):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    # 根据阈值构建掩膜
        mask = cv2.inRange(hsv, lowerRGB, upperRGB)
                    # 腐蚀操作
        mask = cv2.erode(mask, None, iterations=2)
                    # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                    # 初始化瓶盖圆形轮廓质心
        center = None
                    # 如果存在轮廓
        if len(cnts) > 0:
                        # 找到面积最大的轮廓
            c = max(cnts, key=cv2.contourArea)
                        # 确定面积最大的轮廓的外接圆
            ((x, y), radius) = cv2.minEnclosingCircle(c)
                        # 计算轮廓的矩
            M = cv2.moments(c)
                        # 计算质心
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        # 只有当半径大于10时，才执行画图
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, word, (int(x), int(y)), font, 1.2, (255, 255, 255), 2)
                result = {}
                result["x"] = x
                result["y"] = y
                
                return result

    def streaming(self):

        try:
            print("Host: ", self.host_name + " " + self.host_ip)
            print("Connection from: ", self.client_address)
            print("Streaming...")
            print("Press 'q' to exit")

            redLower = np.array([170, 100, 200])
            redUpper = np.array([179, 255, 255])

            greenLower = np.array([65, 100, 100])
            greenUpper = np.array([85, 255, 255])

            #blueLower = np.array([0, 0, 150])
            #blueUpper = np.array([100, 100, 255])
            blueLower = np.array([95, 100, 100])
            blueUpper = np.array([115, 255, 255])
            yellowLower = np.array([5, 100, 100])
            yellowUpper = np.array([20, 255, 255])
            # need bytes here
            stream_bytes = b" "
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b"\xff\xd8")
                last = stream_bytes.find(b"\xff\xd9")
                #str_ = 'S270'
                #str_ = str_.encode("utf-8")
                #socket_tcp.send(str_)
                
                #f = open('record_' + str(self.count) + '.json', 'w')
                #json.dump(dic_dump, f)
                #f.close()
                
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    frame = image
                    result_red = self.draw(frame, redLower, redUpper, "RED")
                    result_green = self.draw(frame, greenLower, greenUpper, "GREEN")
                    result_blue = self.draw(frame, blueLower, blueUpper, "blue")
                    result_yellow = self.draw(frame, yellowLower, yellowUpper, "YELLOW")
                    result = {}
                    result["red"] = result_red
                    result["green"] = result_green
                    result["blue"] = result_blue
                    result["yellow"] = result_yellow

                    self.temp_result = result
                    flag = True
                    if not result_red:
                        flag = False
                    if not result_green:
                        flag = False
                    if not result_blue:
                        flag = False
                    if not result_yellow:
                        flag = False
                    if flag:
                        self.get_finnal_result()
                        self.time_now = int((time.time() - start_time)*1000)
                        self.dqn_loop()
                        '''
                        dic_dump = {'data': self.finnal_result, 'time' : self.time_now}
                        f = open('./test_1/record_' + str(self.count) + '.json', 'w')
                        json.dump(dic_dump, f)
                        f.close()
                        self.count +=1
                        '''
                        cv2.line(frame, 
                                    (int(self.temp_result["red"]["x"]), int(self.temp_result["red"]["y"])), 
                                    (int(self.temp_result["green"]["x"]), int(self.temp_result["green"]["y"])),
                                    (0, 255, 0),
                                    1,
                                    4
                                )
                        cv2.line(frame, 
                                    (int(self.temp_result["blue"]["x"]), int(self.temp_result["blue"]["y"])), 
                                    (int(self.temp_result["yellow"]["x"]), int(self.temp_result["yellow"]["y"])),
                                    (0, 255, 0),
                                    1,
                                    4
                                )
                        cv2.line(frame,
                                    (int(self.finnal_result["me"]["x0"]), int(self.finnal_result["me"]["y0"])),
                                    (int(self.WIDTH/2), int(self.WIDTH/2)),
                                    (0, 0, 255),
                                    4,
                                    4
                                    )
                        cv2.line(frame,
                                    (int(self.finnal_result["enemy"]["x0"]), int(self.finnal_result["enemy"]["y0"])),
                                    (int(self.WIDTH/2), int(self.WIDTH/2)),
                                    (255, 0, 0),
                                    4,
                                    4
                                    )
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,  str(self.finnal_result["me"]["alpha_big"]),
                                            (int(self.finnal_result["me"]["x0"]), int(self.finnal_result["me"]["y0"])), 
                                            font, 
                                            1, 
                                            (0, 255, 0), 
                                            1)
                        cv2.putText(frame, str(self.finnal_result["enemy"]["alpha_small"]),
                                            (int(self.finnal_result["enemy"]["x0"]), int(self.finnal_result["enemy"]["y0"])), 
                                            font, 
                                            1, 
                                            (0, 255, 0), 
                                            1)
                    else:
                        str_S = "S360"
                        str_T = "T370"
                        str_S = str_S.encode("utf-8")
                        str_T = str_T.encode("utf-8")
                        socket_tcp.send(str_S)
                        socket_tcp.send(str_T)
                    #print(self.finnal_result)
                    cv2.imshow("Frame", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.connection.close()
            self.server_socket.close()


if __name__ == "__main__":
    # host, port
    h, p = "192.168.137.1", 8000
    #h = input("the up pi ip: ")
    VideoStreamingTest(h, p)
