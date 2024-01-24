import os

import torch
import cv2
import numpy as np
import pywt
import scipy


# 快速傅里叶
def fft(x, rate):
    # the smaller rate, the smoother; the larger rate, the darker
    # rate = 4, 8, 16, 32
    mask = torch.zeros(x.shape)
    w, h = x.shape[-2:]
    line = int((w * h * rate) ** .5 // 2)
    mask[:, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1

    fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
    # mask[fft.float() > self.freq_nums] = 1
    # high pass: 1-mask, low pass: mask
    fft = fft * (1 - mask)
    # fft = fft * mask
    fr = fft.real
    fi = fft.imag

    fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    inv = torch.fft.ifft2(fft_hires, norm="forward").real

    inv = torch.abs(inv)

    return inv

# 小波变换
def wavelet(x):
    LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')

    LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255
    HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255
    HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255

    merge1 = HL + LH + HH
    merge1 = (merge1 - merge1.min()) / (merge1.max() - merge1.min()) * 255
    merge1 = cv2.resize(merge1, [x.shape[1], x.shape[0]])

    return merge1


if __name__ == '__main__':
    # image_path = "G:/dataset/Dataset_BUSI_original/Dataset_BUSI_with_GT/malignant/malignant (1).png"
    # img = cv2.imread(image_path)
    # print(img.shape)
    # img = img.transpose(2, 1, 0)
    # img = torch.tensor(img)
    # print(img.shape)
    # inv = fft(img, 0.15)
    # inv = np.array(inv)
    # inv = inv.transpose(2,1,0)
    # cv2.imwrite("G:/malignant (1)_FFT.png", inv*255)
    # print(inv.shape)

    # source_path = "G:/dataset/SAMUS_dataset/BUSI/Breast-BUSI/img"
    # save_path = "G:/dataset/SAMUS_dataset/BUSI/Breast-BUSI/H1"
    # for i in os.listdir(source_path):
    #     img = cv2.imread(os.path.join(source_path, i), cv2.IMREAD_GRAYSCALE)
    #     img = np.array(img)
    #     print(img.shape)
    #     lf = wavelet(img)
    #     print(lf.shape)
    #     cv2.imwrite(os.path.join(save_path, i), lf)

    path = "G:/dataset/SAMUS_dataset/BUSI/Breast-BUSI/img/benign0026.png"
    img = cv2.imread(path, 0)
    img = np.array(img)
    lf = wavelet(img)
    cv2.imwrite("G:/benign0026_H.png", lf)


