import json
import requests
import cv2
import numpy as np
import time
import os
import shutil
from multiprocessing import Pool
import sys
import csv

from pathlib import Path


class DatasetProcess(object):
    def __init__(self):
        self.frame_rate = 10
        self.label_path = "/mnt/zhaoqingsong/data/baidu_data/dataset_output_mark"
        self.save_json_path = "/mnt/zhaoqingsong/data/baidu_data/json_url"
        self.download_jpg_path = "/mnt/zhaoqingsong/data/baidu_data/down_jpg"
        self.bad_seq_path = "/mnt/zhaoqingsong/data/baidu_data/bad_seq"
        self.json_log = "/mnt/zhaoqingsong/data/baidu_data/{:}.json"
        pass

    def cut_seq(self):
        file_list = os.listdir(self.download_jpg_path)
        file_list.sort()
        # cut the file whose jpg number under 10
        for i_cou, i_con in enumerate(file_list):
            temp_file_path = os.path.join(self.download_jpg_path, i_con)
            if len(os.listdir(temp_file_path)) < 10:
                dst_path = self.bad_seq_path
                try:
                    shutil.move(temp_file_path, dst_path)
                    pass
                except ValueError:
                    print("\'{:}\' something wrong!!!!".format(i_con))
                    continue
                    pass
                pass
            pass
        pass

    def clear_sequence(self):
        file_list = os.listdir(self.download_jpg_path)
        file_list.sort()
        save_json = []
        # cut the file whose sequence not continuous
        for i_cou, i_con in enumerate(file_list):
            first_file_path = i_con.split('_')[0] + '.json'
            second_file_path = int(i_con.split('_')[1])
            json_file_path = os.path.join(self.save_json_path, first_file_path)
            temp_file_path = os.path.join(self.download_jpg_path, i_con)
            temp_json = self.read_json(json_file_path)[second_file_path]
            length_json = len(temp_json)
            temp_question = [True] * (length_json-1)
            for i in range(1, length_json):
                interval = int(os.path.basename(temp_json[i])[:-4].split('_')[0]) - int(os.path.basename(temp_json[i-1])[:-4].split('_')[0])
                if (interval//1000000 > 105) or (interval//1000000 < 95):
                    temp_log = "{:}: {:}: {:} sub {:} is {:}".format(i_con,
                                                                     i,
                                                                     int(os.path.basename(temp_json[i])[:-4].split('_')[0]),
                                                                     int(os.path.basename(temp_json[i-1])[:-4].split('_')[0]),
                                                                     interval//1000000)
                    save_json.append(temp_log)

                    print("{:}: {:}: {:} sub {:}".format(i_con,
                                                         i,
                                                         int(os.path.basename(temp_json[i])[:-4].split('_')[0]),
                                                         int(os.path.basename(temp_json[i-1])[:-4].split('_')[0])),
                          '\t',
                          interval//1000000)
                    temp_question[i-1] = False
                pass

            if False in temp_question:
                dst_path = self.bad_seq_path
                try:
                    shutil.move(temp_file_path, dst_path)
                    print("\'{:}\' has moved~".format(i_con))
                    pass
                except ValueError:
                    print("\'{:}\' something wrong!!!!".format(i_con))
                    continue
                    pass
                pass
            pass
        save_json_path = self.json_log.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        self.store_json(save_json_path, save_json)
        pass

    @staticmethod
    def number2time(input_number=1607410456588144000):
        assert len(str(input_number)) == 19, "{:} input number error!".format(input_number)

        us = input_number // 1000
        ms = us // 1000
        sec = ms // 1000
        minutes = sec // 60
        hours = minutes // 60
        days = hours // 24
        # print(days)
        ns = input_number % 1000
        us = us % 1000
        ms = ms % 1000
        sec = sec % 60
        minutes = minutes % 60
        hours = hours % 24

        return [days, hours, minutes, sec, ms, us, ns]
        pass

    @staticmethod
    def store_json(store_path, data):
        with open(store_path, 'w') as json_file:
            json_file.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False))
            pass
        pass

    @staticmethod
    def read_json(data):
        with open(data, 'r') as load_f:
            load_dict = json.load(load_f)
            pass
        return load_dict
        pass

    @staticmethod
    def write_csv(load_json, csv_path):
        f = open(csv_path, 'w')
        csv_write = csv.writer(f)
        csv_write.writerow(load_json.keys())
        for key in load_json.keys():
            csv_write.writerow(load_json[key])
            pass
        f.close()
        pass

    @staticmethod
    def is_path_existed_if_no_mk_it(path):
        """
        Check the path existing or not, if not create it.
        :param path: Must be a path not a file.
        :return: No return.
        """
        if not os.path.exists(path):
            os.mkdir(path)
            pass
        pass

    def read_txt_and_download(self, path):
        path = os.path.join(self.label_path, path)
        split_character = "\t"
        self.is_path_existed_if_no_mk_it(self.download_jpg_path)
        with open(path, encoding='UTF-8') as fb:
            f_content_url = []
            # f_content_file_name = []
            for line_count, line in enumerate(fb.readlines()[1:]):
                f_line = line.strip("\n").split(split_character)
                assert f_line.__len__() == 3, "{:} label data format error!".format(os.path.basename(path))
                temp_url_line = json.loads(f_line[0])
                f_content_url.append(temp_url_line)

                # download 'jpg' into my computer
                temp_file_name = os.path.basename(path)[:-4] + '_' + str(line_count)
                temp_file_name = os.path.join(self.download_jpg_path, temp_file_name)
                self.is_path_existed_if_no_mk_it(temp_file_name)

                for i_cou, i_con in enumerate(temp_url_line):
                    temp_url_jpg = self.number2time(int(os.path.basename(i_con)[:-4].split('_')[0]))
                    temp_url_jpg = "{:d}_{:02d}_{:02d}_{:02d}_{:03d}.jpg".format(temp_url_jpg[0], temp_url_jpg[1],
                                                                                 temp_url_jpg[2], temp_url_jpg[3],
                                                                                 temp_url_jpg[4], temp_url_jpg[5])

                    # print(temp_file_name, '\n', temp_url_jpg)
                    if not os.path.exists(os.path.join(temp_file_name, temp_url_jpg)):
                        try:
                            if self.get_file(path=temp_file_name, file_name=temp_url_jpg, url=i_con):
                                # print("\'{:}\' has download and named to \'{:}\'.".format(os.path.basename(i_con),
                                #                                                          temp_url_jpg))
                                print("\'{:}\' done!.".format(temp_url_jpg))
                                pass
                            pass
                        except ValueError:
                            print("\'{:}\' download failed!!!!!!!!!!!!!!!".format(os.path.basename(i_con)))
                            continue
                            pass
                    # exit()
                    pass
                # f_content_file_name.append(json.loads(f_line[1]))
                pass

            # save url into json file.
            _save_json_path = os.path.join(self.save_json_path, os.path.basename(path)[:-3] + 'json')
            if not os.path.exists(_save_json_path):
                self.store_json(_save_json_path, f_content_url)
                print("\'{:}\' url has turn to {:}json.".format(os.path.basename(path), os.path.basename(path)[:-3]))
                pass
            pass
        return 1
        pass

    @staticmethod
    def old_read_txt(path):
        with open(path, 'r') as fb:
            txt = {}
            datas = []
            read = fb.read().splitlines()
            keys = read[0].split('\t')
            lines = read[1:]
            for line in lines:
                data = line.split('\t')
                datas.append(data)

            for i, key in enumerate(keys):
                txt[key] = []
                for data in datas:
                    txt[key].append(data[i])

        for key in txt.keys():
            if key != '最终答案':
                new_data = []
                for ori_data in txt[key]:
                    ori_data_jsons = json.loads(ori_data)
                    for ori_data_json in ori_data_jsons:
                        new_data.append(ori_data_json)
                txt[key] = new_data

            if key == '最终答案':
                new_data = []
                for ori_data in txt[key]:
                    ori_data_jsons = json.loads(ori_data)['result']
                    for ori_data_json in ori_data_jsons:
                        new_json = {}
                        new_json['elements'] = ori_data_json['elements']
                        new_json['size'] = ori_data_json['size']
                        new_data.append(new_json)
                txt[key] = new_data
        return txt

    @staticmethod
    def get_file(path, file_name, url):
        r = requests.get(url)
        with open(Path(path, file_name), "wb") as code:
            code.write(r.content)
            pass
        return True
        pass

    @staticmethod
    def old_create_label(path, json, file_name):
        size = np.array([json['size']['height'], json['size']['width']])
        img = np.zeros(size)
        for points in json['elements']:
            if points['text'] == '同向区域':
                point_np = np.asarray([[x['x'], x['y']] for x in points['points']])
                vertices = np.array([point_np], dtype=np.int32)
                cv2.fillPoly(img, vertices, color=255)
        cv2.imwrite(str(Path(path, file_name)), img)

    def old_data_process(self, PATH):
        txt_list = Path(PATH).iterdir()
        txt_list = [i for i in txt_list]
        print(txt_list)
        for i, txt_path in enumerate(txt_list):
            print('prepare data:' + str(i) + '/' + str(len(txt_list)))
            txt = self.read_txt(txt_path)
            for txt_line in range(len(txt['url'])):
                if txt_line % 4 != 3:
                    self.get_file(path='./data/list/baidu/img/train', file_name=txt['file_name'][txt_line],
                                  url=txt['url'][txt_line])
                    self.create_label(path='./data/list/baidu/label/train', json=txt['最终答案'][txt_line],
                                      file_name=txt['file_name'][txt_line])

                else:
                    self.get_file(path='./data/list/baidu/img/val', file_name=txt['file_name'][txt_line],
                                  url=txt['url'][txt_line])
                    self.create_label(path='./data/list/baidu/label/val', json=txt['最终答案'][txt_line],
                                      file_name=txt['file_name'][txt_line])

    @staticmethod
    def old_write_lst(path):

        def get_Str(path, img_name):
            return str(Path(path, img_name))

        img_path = Path(path, 'img')
        label_path = Path(path, 'label')

        train_img_path = Path(img_path, 'train')
        val_img_path = Path(img_path, 'val')
        test_img_path = Path(img_path, 'test')

        train_label_path = Path(label_path, 'train')
        val_label_path = Path(label_path, 'val')
        test_label_path = Path(label_path, 'test')

        with open('data/list/baidu/train1024.lst', 'w') as f:
            img_names = train_img_path.iterdir()
            for img_name in img_names:
                f.write(get_Str('img/train', img_name.name) + '\t' + get_Str('label/train', img_name.name) + '\n')

        with open('data/list/baidu/val.lst', 'w') as f:
            img_names = val_img_path.iterdir()
            for img_name in img_names:
                f.write(get_Str('img/val', img_name.name) + '\t' + get_Str('label/val', img_name.name) + '\n')

        with open('data/list/baidu/test.lst', 'w') as f:
            img_names = val_img_path.iterdir()
            for img_name in img_names:
                f.write(get_Str('img/val', img_name.name) + '\n')
        pass

    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    my_task = DatasetProcess()
    print('start ...')

    t1 = time.time()

    # clear dataset
    my_task.clear_sequence()

    # _label_path = "/Users/zhaoqingsong/my-pycharm/self_supervised_data_preprocess/dataset_output_mark/"
    # label_list = os.listdir(my_task.label_path)
    # label_list.sort()
    # print("Total: ", label_list.__len__())

    # --loop realizing--
    # for i_count, i in enumerate(label_list):
    #     temp_txt_path = os.path.join(my_task.label_path, i)
    #     my_task.read_txt_and_download(temp_txt_path)
    #     pass

    # --Multiple processes--
    # p = Pool(12)
    # p.map(my_task.read_txt_and_download,  label_list)
    # p.close()
    # p.join()

    t2 = time.time()
    print('take time:' + str(t2 - t1) + 's' + 'end.')

    pass

"""
url = "https://zhongce-yy.bj.bcebos.com/赢彻图像数据/united-discrete/2020/12/18/0d40b6fd-2767-402c-88a9-ef63bd1877d4/spherical-left-backward/1607410463187883000_1607410463186283808.jpg"
    file_name = "test1.jpg"
    my_task.get_file(path, file_name, url)
    
# numbers = [1607323583787635000, 1607323583800156552, 1607323583887831000, 1607323583900181648, 1607323584887799000, 1607323584900432160]
    # times = my_task.number2time(numbers[0])
    # print("Time is {:} D: {:02d} H: {:02d} M: {:02d} S: {:03d} Ms: {:03d} Us".format(times[0], times[1], times[2],
    #                                                                                  times[3], times[4], times[5]))
    # times = my_task.number2time(numbers[1])
    # print("Time is {:} D: {:02d} H: {:02d} M: {:02d} S: {:03d} Ms: {:03d} Us".format(times[0], times[1], times[2],
    #                                                                                  times[3], times[4], times[5]))
    # times = my_task.number2time(numbers[2])
    # print("Time is {:} D: {:02d} H: {:02d} M: {:02d} S: {:03d} Ms: {:03d} Us".format(times[0], times[1], times[2],
    #                                                                                  times[3], times[4], times[5]))
    # times = my_task.number2time(numbers[3])
    # print("Time is {:} D: {:02d} H: {:02d} M: {:02d} S: {:03d} Ms: {:03d} Us".format(times[0], times[1], times[2],
    #                                                                                  times[3], times[4], times[5]))
    # times = my_task.number2time(numbers[4])
    # print("Time is {:} D: {:02d} H: {:02d} M: {:02d} S: {:03d} Ms: {:03d} Us".format(times[0], times[1], times[2],
    #                                                                                  times[3], times[4], times[5]))
    # times = my_task.number2time(numbers[5])
    # print("Time is {:} D: {:02d} H: {:02d} M: {:02d} S: {:03d} Ms: {:03d} Us".format(times[0], times[1], times[2],
    #                                                                                  times[3], times[4], times[5]))
    # exit()
    # label_path = "/Users/zhaoqingsong/my-pycharm/self_supervised_data_preprocess/dataset_output_mark/345572.txt"  # Users/zhaoqingsong/my-pycharm/self_supervised_data_preprocess/
    # my_txt = my_task.read_txt(label_path)
    # print(my_txt)
    # exit()
#x1024=0
#x1080=0
#PATH='./data/list/baidu/img/train'
#imgs=Path(PATH).iterdir()
#for img in imgs:
    #image=cv2.imread(str(img),cv2.IMREAD_COLOR)
    #label=cv2.imread(str(Path(Path(PATH).parent.parent,'label','train',img.name)),cv2.IMREAD_GRAYSCALE)
    #if image.shape[0]!=label.shape[0] or image.shape[1]!=image.shape[1]:
        #print(image.shape,label.shape)
# data_process('data/list/baidu/output-mark__20210226150649')
    write_lst('data/list/baidu')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

"""