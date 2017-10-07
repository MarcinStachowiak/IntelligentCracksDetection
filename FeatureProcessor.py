import glob
import os
import shutil
import numpy as np

from ExportService import ExportService
from ImageProcessingService import ImageProcessingService
from ImportService import ImportService
from PropertiesKeys import PropertiesKeys
from PropertiesSupport import PropertiesSupport
from FeatureService import FeatureService


class FeatureProcessor:
    @staticmethod
    def read_split_and_pre_process_all_images(pixel_is_feature=False):
        prop = PropertiesSupport.load_properties_img_processing()
        file_with_features_path = prop[PropertiesKeys.FILE_WITH_FEATURES]
        source_folder = prop[PropertiesKeys.SPLITTED_PROCESSED_IMAGES_TARGET_FOLDER]
        first_loop = True

        if os.path.isfile(file_with_features_path):
            os.remove(file_with_features_path)
            print("Removal of %s" % file_with_features_path)

        folders = os.listdir(source_folder)
        print("Found %d folders (classes)" % len(folders))

        for folder_name in folders:
            class_id = FeatureProcessor.determine_class_number(folder_name)
            path_to_class = os.path.join(source_folder, folder_name)
            all_features_for_class = FeatureProcessor.compute_features_for_class(path_to_class, class_id,pixel_is_feature)
            if first_loop:
                first_loop = False
                all_features = np.empty((0, all_features_for_class.shape[1]), int)
            all_features = np.vstack([all_features, all_features_for_class])

        ExportService.save_array_to_csv(all_features, file_with_features_path)
        print("Saved %d rows to %s" % (all_features.shape[1], file_with_features_path))

        print("Computing features finished with success")

    @staticmethod
    def compute_features_for_class(path_to_folder, class_id,pixel_is_feature=False):
        print("Starting features computing for images located in %s" % path_to_folder)
        if os.path.isdir(path_to_folder):
            file_search_string = os.path.join(path_to_folder, "*" + ".jpg")
            jpg_files = glob.glob(file_search_string)
            print("Wczytano %d zdjęć" % len(jpg_files))
            first_loop = True

            for image_file in jpg_files:
                print("Features computing for image %s" % image_file)
                image_data = ImportService.read_simple_image(image_file)
                features_vector = FeatureService.calculatepe_feature_vector(image_data,pixel_is_feature)
                features_vector = np.append(features_vector, [class_id])
                if first_loop:
                    first_loop = False
                    all_features = np.empty((0, features_vector.shape[0]), int)
                all_features = np.vstack([all_features, features_vector])
        else:
            raise NotADirectoryError("Path does not exist or is not a directory.")
        return (all_features)

    @staticmethod
    def determine_class_number(class_name):
        if class_name == "Crack Field":
            result = 0
        elif class_name == "Crack-Like":
            result = 1
        elif class_name == "Geometry":
            result = 2
        elif class_name == "Irrelevant":
            result = 3
        elif class_name == "Lamination":
            result = 4
        elif class_name == "Mill Anomaly":
            result = 5
        elif class_name == "Notch-Like":
            result = 6
        elif class_name == "Weld Imperfection":
            result = 7
        else:
            raise NotADirectoryError("Wrong class (folder) name %s" % class_name)
        return (result)
