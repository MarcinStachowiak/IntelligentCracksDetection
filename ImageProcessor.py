import glob
import os

from ExportService import ExportService
from ImageProcessingService import ImageProcessingService
from ImportService import ImportService
from PropertiesKeys import PropertiesKeys
from PropertiesSupport import PropertiesSupport
from SplitService import SplitService


class ImportProcessor:
    @staticmethod
    def read_split_and_pre_process_images_for_class(path_to_folder, with_original_save=False):
        """
        AF_ReadSplitAndPreProcessImagesForClass
        :param path_to_folder: Absolute path to a folder with images
        :param with_original_save: The flag determines whether original split images should be saved
        :return: 3d array with all images
        """
        if os.path.isdir(path_to_folder):
            file_search_string = os.path.join(path_to_folder, "*" + ".png")
            jpg_files = glob.glob(file_search_string)
            print("Wczytano %d zdjęć" % len(jpg_files))
            current_folder_name = os.path.basename(os.path.normpath(path_to_folder))

            for image_file in jpg_files:
                file_name = os.path.splitext(os.path.basename(image_file))[0]
                image_data = ImportService.read_simple_image(os.path.join(path_to_folder, image_file))
                prop = PropertiesSupport.load_properties_img_processing()
                splitted_image_height = int(prop[PropertiesKeys.SPLITTED_IMAGE_HEIGHT])
                split_image_data = SplitService.split_crack_image_to_sensors(image_data, splitted_image_height)
                if with_original_save:
                    target_original_images = prop[PropertiesKeys.SPLITTED_ORIGINAL_IMAGES_TARGET_FOLDER]
                    ExportService.save_images(split_image_data, file_name,
                                              os.path.join(target_original_images, current_folder_name))
                pre_processed_image_data = ImageProcessingService.pre_process_image(split_image_data)
                target_processed_images = prop[PropertiesKeys.SPLITTED_PROCESSED_IMAGES_TARGET_FOLDER]
                ExportService.save_images(pre_processed_image_data, file_name,
                                          os.path.join(target_processed_images, current_folder_name))
        else:
            raise NotADirectoryError("Path does not exist or is not a directory.")


ImportProcessor.read_split_and_pre_process_images_for_class(
    "C:\\Users\\Marcin\\Downloads\\comp\\sample_train_200x20_all\\Crack Field", True)
