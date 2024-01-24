import os
import shutil


def rename(img_name):
    img_name = img_name.split('.')[0]
    img_name_list = list(img_name)
    pos1 = img_name.index('(')
    pos2 = img_name.index(')')
    numlen = pos2 - pos1 - 1
    if numlen == 1:
        img_name_list.insert(pos1 + 1, '000')
    elif numlen == 2:
        img_name_list.insert(pos1 + 1, '00')
    elif numlen == 3:
        img_name_list.insert(pos1 + 1, '0')
    img_name = "".join(img_name_list)

    c_to_remove = " ()"
    for c in c_to_remove:
        img_name = img_name.replace(c, '')
    return img_name


def save(dataset_path, name, img_savepath, label_savepath):
    data_path = os.path.join(dataset_path, name)
    label_list = os.listdir(data_path)
    for label in label_list:
        img_list = os.listdir(os.path.join(data_path, label))
        for img in img_list:
            if "_mask" not in img:
                img_name = rename(img)
                shutil.copy(os.path.join(data_path, label, img), os.path.join(img_savepath, img_name+".png"))
            else:
                img_name = rename(img)
                label_name = img_name.replace("_mask", "")
                shutil.copy(os.path.join(data_path, label, img), os.path.join(label_savepath, label_name+".png"))
    print("%s dataset save done" % name)


if __name__ == '__main__':
    dataset_path = "G:/dataset/breast_BUSI/"
    source_path = "G:/dataset/SAMUS/Breast-BUSI/"
    img_savepath = os.path.join(source_path, "img")
    label_savepath = os.path.join(source_path, "label")

    save(dataset_path, "train", img_savepath, label_savepath)
    save(dataset_path, "val", img_savepath, label_savepath)
    save(dataset_path, "test", img_savepath, label_savepath)
