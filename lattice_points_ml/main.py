
import csv
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


def main():
    annot_train_filename = 'lattice_points_train_annot.csv'
    annot_test_filename = 'lattice_points_test_annot.csv'

    model: ConvNet = ConvNet()
    model = model.to(model.device)
    model.load_model(os.getcwd() + '\\lattice_points_ml\\model\\model_1.pt')
    # model.test_model(annot_test_filename)
    model.predict_model(cv2.imread(os.getcwd() + '\\lattice_points_ml\\latchess21\\ok_test\\65100926823770869_270.jpg', cv2.IMREAD_GRAYSCALE))

if __name__ == '__main__':
    main()
    # folder_from = os.getcwd() + '\\lattice_points_ml\\latchess21\\ok_test1'
    # folder_to = os.getcwd() + '\\lattice_points_ml\\latchess21\\ok_test'
    # resave_to_gray_image(folder_from, folder_to)
