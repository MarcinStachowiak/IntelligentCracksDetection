import glob
import os

from ExportService import ExportService
from ImageProcessingService import ImageProcessingService
from ImportService import ImportService
from PropertiesKeys import PropertiesKeys
from PropertiesSupport import PropertiesSupport
from SplitService import SplitService
import shutil


class ImageProcessor:
    @staticmethod
    def read_split_and_pre_process_all_images(path_to_source_dir, with_original_save):
        prop = PropertiesSupport.load_properties_img_processing()
        target_original_images = os.path.join(os.getcwd(), prop[PropertiesKeys.SPLITTED_ORIGINAL_IMAGES_TARGET_FOLDER])
        target_processed_images = os.path.join(os.getcwd(),
                                               prop[PropertiesKeys.SPLITTED_PROCESSED_IMAGES_TARGET_FOLDER])
        if os.path.isdir(target_original_images):
            shutil.rmtree(target_original_images)
            print("Removal of %s" % target_original_images)
        if os.path.isdir(target_processed_images):
            shutil.rmtree(target_processed_images)
            print("Removal of %s" % target_processed_images)
        folders = os.listdir(path_to_source_dir)
        print("Found %d folders (classes)" % len(folders))

        for folder_name in folders:
            path_to_class = os.path.join(path_to_source_dir, folder_name)
            ImageProcessor.read_split_and_pre_process_images_for_class(path_to_class, with_original_save)

        print("Processing finished with success")

    @staticmethod
    def read_split_and_pre_process_images_for_class(path_to_folder, with_original_save=False):
        """
        AF_ReadSplitAndPreProcessImagesForClass
        :param path_to_folder: Absolute path to a folder with images
        :param with_original_save: The flag determines whether original split images should be saved
        :return: 3d array with all images
        """
        print("Starting processing images located in %s" % path_to_folder)
        if os.path.isdir(path_to_folder):
            file_search_string = os.path.join(path_to_folder, "*" + ".png")
            jpg_files = glob.glob(file_search_string)
            print("Wczytano %d zdjęć" % len(jpg_files))
            prop = PropertiesSupport.load_properties_img_processing()
            current_folder_name = os.path.basename(os.path.normpath(path_to_folder))

            for image_file in jpg_files:
                file_name = os.path.splitext(os.path.basename(image_file))[0]
                image_data = ImportService.read_simple_image(os.path.join(path_to_folder, image_file))
                splitted_image_height = int(prop[PropertiesKeys.SPLITTED_IMAGE_HEIGHT])
                splitted_image_width = int(prop[PropertiesKeys.SPLITTED_IMAGE_WIDHT])
                split_image_data = SplitService.split_crack_image_to_sensors(image_data, splitted_image_height,
                                                                             splitted_image_width)
                if with_original_save:
                    target_original_images = prop[PropertiesKeys.SPLITTED_ORIGINAL_IMAGES_TARGET_FOLDER]
                    ExportService.save_images(split_image_data, file_name,
                                              os.path.join(os.getcwd(), target_original_images, current_folder_name))
                pre_processed_image_data = ImageProcessingService.pre_process_images(split_image_data)
                target_processed_images = prop[PropertiesKeys.SPLITTED_PROCESSED_IMAGES_TARGET_FOLDER]
                filled_treshold = float(prop[PropertiesKeys.FILLED_TRESHOLD])
                result_image_data = ImageProcessingService.filter_filled_images(pre_processed_image_data,
                                                                                splitted_image_height,
                                                                                splitted_image_width, filled_treshold)
                ExportService.save_images(result_image_data, file_name,
                                          os.path.join(os.getcwd(), target_processed_images, current_folder_name))
        else:
            raise NotADirectoryError("Path does not exist or is not a directory.")