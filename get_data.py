import os
import cv2
from skimage.transform import resize


class Get_Dataset():
    def __init__(self, path, resize_size = False):
        self.path = path
        self.resize = resize_size


    @staticmethod
    def Resize(image, resize_size):
        resized_image = resize(image, (resize_size, resize_size))
        return resized_image


    def train(self):
        path = self.path + '/train'
        images_list = []
        labels_list = []

        for foldername in os.listdir(path):
            folderpath = path + '/' + foldername
            print(f'... Downloading {folderpath} Images ...')

            for indx, img in enumerate(os.listdir(folderpath)):
                filepath  = folderpath + '/' + img

                image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                label = int(foldername)

                if self.resize :
                    image = Get_Dataset.Resize(image, self.resize)
                
                images_list.append(np.array(image))
                labels_list.append(np.array(label))

        return np.array(images_list), np.array(labels_list)


    def test(self):
        path = self.path + '/test'
        images_list = []
        images_names = []

        for indx, img in enumerate(os.listdir(path)):
            filepath = path + '/' +img
            print(f'... Downloading {filepath} Images ...')

            image = cv2.imread(filepath, cv2.IMREAD_COLOR)

            if self.resize :
                image = Get_Dataset.Resize(image, self.resize)

            images_list.append(np.array(image))
            images_names.append(filename)

        return np.array(images_list), images_names
