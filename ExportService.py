import os
import cv2
import numpy as np

class ExportService:
    @staticmethod
    def save_images(images, prefix, path_to_target_folder):
        print("Exporting %d parts for %s images to %s" % (len(images), prefix, path_to_target_folder))
        if not os.path.isdir(path_to_target_folder):
            os.makedirs(path_to_target_folder)
        for i, img in enumerate(images):
            output_path = os.path.join(path_to_target_folder, prefix + "_p%d%s"% (i,".jpg"))
            ExportService.save_simple_image(img, output_path)
        pass

    @staticmethod
    def save_simple_image(img, output_path):
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    @staticmethod
    def save_array_to_csv(data_array, file_path):
        np.savetxt(file_path,data_array,delimiter=";")
