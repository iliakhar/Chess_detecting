import csv
import os
import random
import uuid

import cv2
import glob
from ConvNet import *


def create_annotation(filename: str, path: str, folders_names: list[str]):
    with open(filename, mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        for ind, class_name in enumerate(folders_names):
            files = glob.glob(path + class_name + '\\*.jpg')
            for filename in files:
                file_writer.writerow([filename, ind])


def resave_to_gray_image(folder_from: str, folder_to: str):
    files = glob.glob(folder_from + '\\*.jpg')
    for file in files:
        img: np.ndarray = cv2.imread(file)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filename = folder_to + file[file.rfind('\\'):]
        cv2.imwrite(filename, gray_img)


def rename_images(folder_from: str, folder_to: str):
    files = glob.glob(folder_from + '\\*.jpg')
    for filename in files:
        img = cv2.imread(filename)
        new_filename = folder_to + '\\' + str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(new_filename, img)


def rotate_n_image(number_of_images: int, folder_from: str, folder_to: str):
    files = glob.glob(folder_from + '\\*.jpg')
    for filename in files:
        # filename = files[random.randint(0, len(files) - 1)]
        img = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        new_filename = folder_to + '\\' + str(uuid.uuid4()) + '.jpg'
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(new_filename, img)


def canny_folders(folder_from: str, folder_to: str):
    files = glob.glob(folder_from + '\\*.jpg')
    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        new_filename = folder_to + '\\' + str(uuid.uuid4()) + '.jpg'
        ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                                     cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh1, 20, 30, apertureSize=3)
        cv2.imwrite(new_filename, edges)


def main():
    # folder_from = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\3\\border_180'
    # folder_to = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\3\\border_270'
    # os.mkdir(folder_to)
    # rotate_n_image(-1, folder_from, folder_to)

    #
    # canny_folders(folder_from, folder_to)

    # # # rename_images(folder_from, folder_to)
    # # rotate_n_image(1000, folder_from, folder_to)
    # # resave_to_gray_image(folder_from, folder_to)

    # annot_path: str = getcwd() + '\\lattice_points_rotate_5_annot.csv'
    # folder_path = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\3\\'
    # create_annotation(annot_path, folder_path, ['no', 'ok', 'border'])

    annot_train_filename = 'lattice_points_rotate_5_annot.csv'

    model: ConvNet = ConvNet()
    model = model.to(model.device)
    model.load_model(getcwd() + '\\lattice_points_ml\\model\\lattice_rotate_detect_6_100.pt')
    number_of_epochs = 50
    batch_size = 8
    model.train_model(number_of_epochs, annot_train_filename, 'lattice_rotate_detect_6_150.pt', batch_size)


if __name__ == '__main__':
    main()
    # folder_from = os.getcwd() + '\\lattice_points_ml\\latchess21\\ok_test1'
    # folder_to = os.getcwd() + '\\lattice_points_ml\\latchess21\\ok_test'
    # resave_to_gray_image(folder_from, folder_to)
