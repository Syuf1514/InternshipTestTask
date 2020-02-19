import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
import json
from pycocotools import mask
from skimage import measure

def augmentations(image, image_mask, sequence, size, save_original):
    """Do specified number of augmentations with sequence augmenter.
    Args:
        image: the image to augment
        image_mask: initial binary mask (0 - background, 1 - foreground) 
        sequnce: Sequential object to call for augmentations
        size: number of different augmented images to make
        save_original: boolean, whether or not leave the original image
    Returns:
        list of tuples (new image, new mask (0 - background, 1 - foreground)) for augmented images
    """
    segmap = SegmentationMapOnImage(image_mask, shape=image.shape, nb_classes=2)
    augs = [sequence(image=image, segmentation_maps=segmap) for _ in range(size - int(save_original))]
    if save_original:
        augs = [(image, segmap)] + augs
    augs = [(image, segmap.get_arr_int().astype(np.uint8)) for image, segmap in augs]
    return augs

def mask_to_annotation(image_id, image_mask):
    """Convert image binary mask to json annotation.
    Args:
        image_id: 'image_id' and 'id' fields to write to annotation
        image_mask: binary mask (0 - background, 1 - foreground)
    Returns:
        annotation dict
    """
    ground_truth_binary_mask = image_mask
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
            "id": image_id
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    
    return annotation

def add_to_dataset(dataset, image, image_mask, image_id):
    """Add the image and its annotation to the dataset.
    Args:
        dataset: the dataset to modify
        image: the image to add to the dataset
        image_mask: binary mask (0 - background, 1 - foreground)
        image_id: 'image_id' and 'id' fields to write to annotation
    Returns:
        Nothing
    """
    dataset['images'].append({"license": 0, "file_name": f"{image_id:08}.jpg",
                               "width": 512, "height": 512, "id": image_id})
    dataset['annotations'].append(mask_to_annotation(image_id, image_mask))
