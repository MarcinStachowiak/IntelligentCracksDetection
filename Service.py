import os
import cv2
import numpy as np


class Service:
    def __init__(self, properties_path):
        self.properties = None
        self.load_properties(properties_path)

    def load_properties(self, file_path, sep='=', comment_char='#'):
        """
        Read the file passed as parameter as a properties file.
        """
        props = {}
        with open(file_path, "rt") as f:
            for line in f:
                l = line.strip()
                if l and not l.startswith(comment_char):
                    key_value = l.split(sep)
                    key = key_value[0].strip()
                    value = sep.join(key_value[1:]).strip().strip('"')
                    props[key] = value
        self.properties = props

    @staticmethod
    def read_simple_image(path_to_image):
        """
        AF_ReadSimpleImage
         :param path_to_image: Path to an image
         :return: 2d array with an image
        """
        if os.path.isfile(path_to_image):
            return cv2.imread(path_to_image, 0)
        else:
            raise FileNotFoundError("Path does not exist or is not a file.")

    def read_split_and_pre_process_images_for_class(self, path_to_folder, with_original_save=False):
        """
        AF_ReadSplitAndPreProcessImagesForClass
        :param path_to_folder: Absolute path to a folder with images
        :param with_original_save: The flag determines whether original split images should be saved
        :return: 3d array with all images
        """
        if os.path.isdir(path_to_folder):
            for image_file in os.listdir(path_to_folder):
                image_data = self.read_simple_image(os.path.join(path_to_folder, image_file))
                split_image_data = self.split_crack_image_to_sensors(image_data, self.properties["SPLITTED_IMAGE_HEIGHT"])
                if with_original_save:
                    self.save_images(split_image_data,
                                     image_file.split('.')[0],
                                     self.properties["SPLITTED_ORIGINAL_IMAGES_TARGET_FOLDER"] + "/" + os.path.basename(
                                         os.path.normpath(path_to_folder)))
                pre_processed_image_data = self.pre_process_image(split_image_data)
                self.save_images(pre_processed_image_data, image_file.split('.')[0],
                                 self.properties["SPLITTED_PREPROCESSED_IMAGES_TARGET_FOLDER"] + "/" + os.path.basename(
                                     os.path.normpath(path_to_folder)))
        else:
            raise NotADirectoryError("Path does not exist or is not a directory.")

    def pre_process_image(self, split_images_data):
        pass

    @staticmethod
    def split_crack_image_to_sensors(image_data, resulting_height, resulting_width=200, buffer_width=2,
                                     resulting_images_count=18):
        """
        AF_SplitCrackImageToSensors
        :param image_data: 2d array with an image
        :param resulting_height: The height of small, split images
        :param resulting_width: The width of small, split images
        :param buffer_width: Width of buffer columns separating each split image
        :param resulting_images_count: Number of split images
        :return: 3d array with split image
        """
        img_height, img_width = image_data.shape
        img_middle_height = img_height//2

        # Finding first row from the bottom of image with any non-black pixels
        img_down_first = None
        for i in range(img_height, img_middle_height + resulting_height):
            if np.any(image_data[i:] != 0):
                img_down_first = i
                break
        if not img_down_first:
            img_down_first = img_height

        # Finding first row from the middle of image with any non-black pixels
        img_up_first = None
        for i in range(img_middle_height, img_middle_height - resulting_height):
            if np.any(image_data[i:] != 0):
                img_up_first = i
                break
        if not img_up_first:
            img_up_first = img_middle_height

        # Splitting image
        split_images_data = []
        height_begin = img_up_first - resulting_height
        height_end = img_up_first
        for i in range(resulting_images_count // 2):
            width_begin = i * resulting_width + (i+1) * buffer_width
            width_end = (i+1) * resulting_width + (i+1) * buffer_width
            split_images_data.append(image_data[height_begin:height_end, width_begin:width_end])
        height_begin = img_down_first - resulting_height
        height_end = img_down_first
        for i in range(resulting_images_count // 2):
            width_begin = i * resulting_width + (i + 1) * buffer_width
            width_end = (i + 1) * resulting_width + (i + 1) * buffer_width
            split_images_data.append(image_data[height_begin:height_end, width_begin:width_end])
        cv2.imshow("y", image_data)
        for i, img in enumerate(split_images_data):
            cv2.imshow("%d" % i, img)
        key = cv2.waitKey(0)
        return np.dstack(split_images_data)

    def save_images(self, data, prefix, path_to_target_folder):
        pass

    def save_simple_image(self, data, path, extension):
        pass

Service.split_crack_image_to_sensors(Service.read_simple_image(r'C:\Users\KKONDRAT\Desktop\DSC\comp\sample_train_200x20_all\Crack Field\5_39_4813.png'), 210)
