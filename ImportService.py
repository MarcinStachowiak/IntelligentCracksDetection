import os
import cv2


class ImportService:
    @staticmethod
    def read_simple_image(path_to_image):
        """
        AF_ReadSimpleImage
         :param path_to_image: Path to an image
         :return: 2d array with an image
        """
        if os.path.isfile(path_to_image):
            result = cv2.imread(path_to_image, 0)
            print("Reading %s image" % (path_to_image))
            return (result)
        else:
            raise FileNotFoundError("Path %s does not exist or is not a file." % (path_to_image))
