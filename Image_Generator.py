import cv2
import os, random
import numpy as np
from parameter import letters

# # Input data generator
def labels_to_text(labels):     # letters의 index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):      # text를 letters 배열에서의 인덱스 값으로 변환
    return list(map(lambda x: letters.index(x), text))


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 batch_size, downsample_factor, max_text_len=9):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

    ## samples의 이미지 목록들을 opencv로 읽어 저장하기, texts에는 label 저장
    def build_data(self):
        print(self.n, " Image Loading start...")
        for i, img_file in enumerate(self.img_dir):
            img = cv2.imread(self.img_dirpath + img_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0

            self.imgs[i, :, :] = img
            self.texts.append(img_file[0:-4])
        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):      ## index max -> 0 으로 만들기
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                label_length[i] = len(text)

            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)