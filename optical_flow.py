# -*- coding: utf-8 -*-
"""
class: videos
Functions:
1. video to optical flows
2. jpgs to video
3. Real-time display of the BGR value and mouse position of the image
"""
import cv2
import numpy as np
import os
import time
from main import DatasetProcess


class Videos(object):
    def __init__(self):
        self.temp_video_path = r"C:\Users\zqs20\Desktop\test.mp4"
        self.jpg_path = r"C:\Users\zqs20\iCloudDrive\temp\baidu_dataset\down_jpg"
        self.video_path = r"C:\Users\zqs20\iCloudDrive\temp\baidu_dataset\video_path"
        self.flow_path = r"C:\Users\zqs20\iCloudDrive\temp\baidu_dataset\flow_path"
        DatasetProcess.is_path_existed_if_no_mk_it(self.video_path)
        DatasetProcess.is_path_existed_if_no_mk_it(self.flow_path)
        self.fps = 10
        self.high = 1024
        self.width = 1536
        self.video_type = '.avi'
        pass

    def jpg2video(self, input_jpg_path):
        # seting video type
        fps = self.fps
        # cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi
        # cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi
        # cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi
        # cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv
        # cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (self.width, self.high)

        out_video_name = os.path.basename(input_jpg_path) + self.video_type
        out_video_path = os.path.join(self.video_path, out_video_name)

        if not os.path.exists(out_video_path):
            out = cv2.VideoWriter(out_video_path, fourcc, fps, size)
            image_list = os.listdir(input_jpg_path)
            image_list.sort()
            for i_cou, i_con in enumerate(image_list):
                temp_img_path = os.path.join(input_jpg_path, i_con)
                frame = cv2.imread(temp_img_path)
                out.write(frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                pass
            out.release()
            cv2.destroyAllWindows()
            pass

        return out_video_path
        pass

    def video2optical_flow(self, input_video_path):
        _flow_path = os.path.join(self.flow_path, os.path.basename(input_video_path)[:-4])
        DatasetProcess.is_path_existed_if_no_mk_it(_flow_path)

        cap = cv2.VideoCapture(input_video_path)
        frames_num = cap.get(7)

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        for i in range(int(frames_num)-1):
            temp_flow_name = "{:04d}.png".format(i)
            flow_name = os.path.join(_flow_path, temp_flow_name)

            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 9, 3, 5, 1.1, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            if not os.path.exists(flow_name):
                cv2.imwrite(flow_name, rgb)
                print("{:d}th frame \'{:}\' has saved~".format(i, temp_flow_name))
                pass
            prvs = next
            pass
        cap.release()
        cv2.destroyAllWindows()
        return 1
        pass

    @staticmethod
    def display_value_gray_img(path):
        img = cv2.imread(path)  # read img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turn to gray

        def on_mouse(event, x, y, flags, param):  # Standard mouse interaction function
            if event == cv2.EVENT_LBUTTONDBLCLK:  # mouse click
                # Display the value of the pixel where the mouse is located
                # Pay attention ! The difference in pixel representation method and coordinate position
                print("y=", y, "x=", x, img[y, x], "\n")
            if event == cv2.EVENT_MOUSEMOVE:  # mouse move
                print("y=", y, "x=", x, img[y, x], "\n")  #
            pass

        cv2.namedWindow("img")  # set windows
        cv2.setMouseCallback("img", on_mouse)
        while True:  # 无限循环
            cv2.imshow("img", img)  # 显示图像
            if cv2.waitKey() == 27:
                cv2.destroyAllWindows()  # 关闭窗口
                break
                pass
            pass
        pass
    pass


if __name__ == "__main__":
    my_task = Videos()
    print('start ...')
    t1 = time.time()
    # mouse display_vaule_gray_img
    img_path = '/Users/zhaoqingsong/Downloads/aachen_000115_000019/aachen_000115_000019_gtFine_labelTrainIds.png'
    my_task.display_value_gray_img(img_path)
    t2 = time.time()
    print('take time:' + str(t2 - t1) + 's' + '\nend.')
    # jpg_list = os.listdir(my_task.jpg_path)
    # jpg_list.sort()
    # random = np.random.randint(0, 100, 1)
    # out_video_path = my_task.jpg2video(os.path.join(my_task.jpg_path, jpg_list[random[0]]))
    # my_task.video2optical_flow(out_video_path)

    pass
