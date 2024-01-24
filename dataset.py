import os
import shutil


def text_create(desktop_path, name):
    desktop_path = desktop_path  # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'
    file = open(full_path, 'w')


def rename_BUSI(img_name):
    img_name = img_name.split('.')[0]
    img_name_list = list(img_name)
    pos = img_name.index('(')
    numlen = len(img_name_list) - pos - 2
    if numlen == 1:
        img_name_list.insert(pos + 1, '000')
    elif numlen == 2:
        img_name_list.insert(pos + 1, '00')
    elif numlen == 3:
        img_name_list.insert(pos + 1, '0')
    img_name = "".join(img_name_list)

    c_to_remove = " ()"
    for c in c_to_remove:
        img_name = img_name.replace(c, '')
    return img_name


def rename_UDIAT(img_name):
    img_name = img_name.split('.')[0]
    id = img_name.split('_')[0]
    class_name = img_name.split('_')[1]
    img_name = class_name.lower() + id[2:]

    return img_name


def busi_text_write(desktop_path, name, dataset_path):
    full_path = desktop_path + name + '.txt'
    with open(full_path, mode='w') as f:
        label_path = os.path.join(dataset_path, name)
        label_list = os.listdir(label_path)
        for label in label_list:
            img_list = os.listdir(os.path.join(label_path, label))
            for img in img_list:
                if "_mask" not in img:
                    img_name = rename_BUSI(img)
                    f.write(img_name + '\n')
    print("%s.txt write done!" % name)

def udiat_text_write(desktop_path, name, dataset_path):
    full_path = desktop_path + name + '.txt'
    with open(full_path, mode='w') as f:
        label_path = os.path.join(dataset_path, name)
        label_list = os.listdir(label_path)
        for label in label_list:
            img_list = os.listdir(os.path.join(label_path, label))
            for img in img_list:
                if "_mask" not in img:
                    img_name = rename_UDIAT(img)
                    f.write(img_name + '\n')
    print("%s.txt write done!" % name)


def text_write_busi(desktop_path, name, dataset_path):
    full_path = desktop_path + name + '-Breast-BUSI.txt'
    with open(full_path, mode='w') as f:
        label_path = os.path.join(dataset_path, name)
        label_list = os.listdir(label_path)
        for label in label_list:
            img_list = os.listdir(os.path.join(label_path, label))
            for img in img_list:
                if "_mask" not in img:
                    img_name = rename_BUSI(img)
                    if name != "test":
                        f.write('1/Breast-BUSI/' + img_name + '\n')
                    else:
                        f.write('Breast-BUSI/' + img_name + '\n')
    print("%s.txt write done!" % name)

def text_write_udiat(desktop_path, name, dataset_path):
    full_path = desktop_path + name + '-Breast-UDIAT.txt'
    with open(full_path, mode='w') as f:
        label_path = os.path.join(dataset_path, name)
        label_list = os.listdir(label_path)
        for label in label_list:
            img_list = os.listdir(os.path.join(label_path, label))
            for img in img_list:
                if "_mask" not in img:
                    img_name = rename_UDIAT(img)
                    if name != "test":
                        f.write('1/Breast-UDIAT/' + img_name + '\n')
                    else:
                        f.write('Breast-UDIAT/' + img_name + '\n')
    print("%s.txt write done!" % name)


def main_BUSI():
    dataset_path = "G:/dataset/SAMUS_dataset/BUSI/breast_BUSI/"  # 已划分好的数据集所在位置

    source_path = "G:/dataset/SAMUS_dataset/BUSI/"
    busidata_path = os.path.join(source_path, "Breast-BUSI")
    # 创建Breast_BUSI/MainPatient中的text/train/val文本文件
    busiMP_path = os.path.join(busidata_path, "MainPatient/")
    if not os.path.exists(busiMP_path):
        os.makedirs(busiMP_path)
    text_create(busiMP_path, "train", )
    text_create(busiMP_path, "val")
    text_create(busiMP_path, "test")

    # 将不同数据集写入相应txt文件中
    busi_text_write(busiMP_path, "train", dataset_path)
    busi_text_write(busiMP_path, "val", dataset_path)
    busi_text_write(busiMP_path, "test", dataset_path)

    # 创建与breast_BUSI数据集同级的MainPatient和其下的txt
    MP_path = os.path.join(source_path, "MainPatient/")
    if not os.path.exists(MP_path):
        os.makedirs(MP_path)
    text_create(MP_path, "train-Breast-BUSI")
    text_create(MP_path, "val-Breast-BUSI")
    text_create(MP_path, "test-Breast-BUSI")

    # 将不同数据集写入相应txt文件中
    text_write_busi(MP_path, "train", dataset_path)
    text_write_busi(MP_path, "val", dataset_path)
    text_write_busi(MP_path, "test", dataset_path)

    # 复制MainPatient中的text/train/val文本文件
    shutil.copy(os.path.join(MP_path, "train-Breast-BUSI.txt"), os.path.join(MP_path, "train.txt"))
    shutil.copy(os.path.join(MP_path, "val-Breast-BUSI.txt"), os.path.join(MP_path, "val.txt"))
    shutil.copy(os.path.join(MP_path, "test-Breast-BUSI.txt"), os.path.join(MP_path, "test.txt"))


def main_UDIAT():
    dataset_path = "G:/dataset/SAMUS_dataset/UDIAT/breast_UDIAT/"  # 已划分好的数据集所在位置

    source_path = "G:/dataset/SAMUS_dataset/UDIAT/"
    udiatdata_path = os.path.join(source_path, "Breast-UDIAT")
    # 创建Breast_BUSI/MainPatient中的text/train/val文本文件
    udiatMP_path = os.path.join(udiatdata_path, "MainPatient/")
    if not os.path.exists(udiatMP_path):
        os.makedirs(udiatMP_path)
    text_create(udiatMP_path, "train", )
    text_create(udiatMP_path, "val")
    text_create(udiatMP_path, "test")

    # 将不同数据集写入相应txt文件中
    udiat_text_write(udiatMP_path, "train", dataset_path)
    udiat_text_write(udiatMP_path, "val", dataset_path)
    udiat_text_write(udiatMP_path, "test", dataset_path)

    # 创建与breast_BUSI数据集同级的MainPatient和其下的txt
    MP_path = os.path.join(source_path, "MainPatient/")
    if not os.path.exists(MP_path):
        os.makedirs(MP_path)
    text_create(MP_path, "train-Breast-UDIAT")
    text_create(MP_path, "val-Breast-UDIAT")
    text_create(MP_path, "test-Breast-UDIAT")

    # 将不同数据集写入相应txt文件中
    text_write_udiat(MP_path, "train", dataset_path)
    text_write_udiat(MP_path, "val", dataset_path)
    text_write_udiat(MP_path, "test", dataset_path)

    # 复制MainPatient中的text/train/val文本文件
    shutil.copy(os.path.join(MP_path, "train-Breast-UDIAT.txt"), os.path.join(MP_path, "train.txt"))
    shutil.copy(os.path.join(MP_path, "val-Breast-UDIAT.txt"), os.path.join(MP_path, "val.txt"))
    shutil.copy(os.path.join(MP_path, "test-Breast-UDIAT.txt"), os.path.join(MP_path, "test.txt"))


if __name__ == '__main__':
    # main_BUSI()
    main_UDIAT()
