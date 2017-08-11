import os
import cv2
import numpy as np


class SplitService:
    @staticmethod
    def split_crack_image_to_sensors(image_data, resulting_height, resulting_width=200, buffer_width=2,
                                     resulting_images_count=18, show=False):
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
        img_middle_height = img_height // 2

        # Finding first row from the bottom of image with any non-black pixels
        img_upper_part_bottom = img_middle_height
        for i in range(img_middle_height, img_height):
            if np.any(image_data[i:] != 0):
                img_down_first = i
                break

        # Finding first row from the middle of image with any non-black pixels
        img_bottom_part_top = img_middle_height
        for i in range(img_middle_height, 0, -1):
            if np.any(image_data[i:] != 0):
                img_up_first = i
                break

        # Splitting image
        split_images_data = []
        height_upper_part_begin = img_upper_part_bottom
        height_upper_part_end = min(img_height, height_upper_part_begin + resulting_height)
        for i in range(resulting_images_count // 2):
            width_begin = i * resulting_width + (i + 1) * buffer_width
            width_end = (i + 1) * resulting_width + (i + 1) * buffer_width
            sensor_image = image_data[height_upper_part_begin:height_upper_part_end, width_begin:width_end]
            sensor_image_form = np.zeros([resulting_height, resulting_width])
            sensor_image_height = sensor_image.shape[0]
            sensor_image_form[0:sensor_image_height, :] = sensor_image_form[0:sensor_image_height, :] + sensor_image
            split_images_data.append(sensor_image_form.astype(np.uint8))

        height_bottom_part_begin = max(0, img_bottom_part_top - resulting_height)
        height_bottom_part_end = img_bottom_part_top
        for i in range(resulting_images_count // 2):
            width_begin = i * resulting_width + (i + 1) * buffer_width
            width_end = (i + 1) * resulting_width + (i + 1) * buffer_width
            sensor_image = image_data[height_bottom_part_begin:height_bottom_part_end, width_begin:width_end]
            sensor_image_form = np.zeros([resulting_height, resulting_width])
            sensor_image_height = sensor_image.shape[0]
            sensor_image_form[0:sensor_image_height, :] = sensor_image_form[0:sensor_image_height, :] + sensor_image
            split_images_data.append(sensor_image_form.astype(np.uint8))

            if show:
                cv2.imshow("y", image_data)
                for i, img in enumerate(split_images_data):
                    cv2.imshow("%d" % i, img)
                    cv2.waitKey(0)

        return np.array(split_images_data)
