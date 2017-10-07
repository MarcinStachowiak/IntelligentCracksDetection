import cv2
import numpy as np

class ImageProcessingService:
    @staticmethod
    def filter_filled_images(images,height,width,threshold=0.2):
        result = []
        for img in images:
            non_zero_count=np.sum(np.any(img,axis=0))
            filled_ratio=non_zero_count/img.shape[1]
            if filled_ratio>=threshold:
                result.append(img)
        return (np.array(result))

    def pre_process_images(images,show=False):
        result = []
        for img in images:
                result.append(ImageProcessingService.pre_process_image(img,show))
        return (np.array(result))

    @staticmethod
    def pre_process_image(split_images_data,show=False):
        if show:
            cv2.imshow("1", split_images_data)

        eq = cv2.equalizeHist(split_images_data)

        if show:
            cv2.imshow("2", eq)

        ret, thresh4 = cv2.threshold(eq, 170, 255, cv2.THRESH_TOZERO)

        if show:
            cv2.imshow("4", thresh4)

        thresh_eq = cv2.equalizeHist(thresh4)

        if show:
            cv2.imshow("4eq", thresh_eq)

        closing = cv2.morphologyEx(thresh_eq, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        if show:
            cv2.imshow("7", closing)

        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))


        if show:
            cv2.imshow("8", opening)

        sobel = cv2.Sobel(opening, cv2.CV_8U, 0, 1, ksize=3)

        if show:
            cv2.imshow("9", sobel)
            cv2.waitKey(0)
        return (sobel)

