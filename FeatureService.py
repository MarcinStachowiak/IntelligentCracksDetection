import cv2
import numpy as np
import mahotas as mh


class FeatureService:
    @staticmethod
    def calculatepe_feature_vectors(img_array):
        first_loop = True
        for i in np.nditer(img_array):
            current_img = img_array[i]
            features_vector = FeatureService.calculatepe_feature_vector(current_img)
            if first_loop:
                first_loop = False
                result = np.empty((0, features_vector.shape[0]), int)
            result = np.vstack([result, features_vector])
        return (result)

    @staticmethod
    def calculatepe_feature_vector(img,pixel_is_feature=False):
        if not pixel_is_feature:
            vector_haralick = mh.features.haralick(img)
            vector_zernike = mh.features.zernike_moments(img, 110, degree=12)
            vector_lbp = mh.features.lbp(img, 4, 8)

            vector_result = np.concatenate(
                [vector_haralick[0, :], vector_haralick[1, :], vector_haralick[2, :], vector_haralick[3, :], vector_zernike,
                vector_lbp])
        else:
            vector_result= img.reshape(210*200,-1)
        return vector_result
