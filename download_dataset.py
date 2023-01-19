from cytomine.models import ImageInstanceCollection, AnnotationCollection
from collections import defaultdict
from cytomine import Cytomine
from shapely import wkt
from PIL import Image

import constants as cst
import numpy as np

import cv2
import os

def download(host, public_key, private_key, id_project, term, term_name, working_path=""):
    with Cytomine(host, public_key, private_key) as cytomine:

        dataset_path = cst.DATA
        images_path = cst.IMG
        mask_path = os.path.join(cst.DATA, term_name)

        # Creating directories
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        # Fetching the annotations
        annotations = AnnotationCollection()
        annotations.project = id_project
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.term = term
        annotations.fetch()

        annot_per_image = defaultdict(list)
        for annot in annotations:
            annot_per_image[annot.image].append(wkt.loads(annot.location))

        # Fetching images
        images = ImageInstanceCollection().fetch_with_filter("project", id_project)

        filepaths = list()
        n_img = 0
        n_empty = 0
        for img in images:
            name = img.originalFilename
            name = name[:-4] + ".jpg"
            filepath = os.path.join(images_path, name)
            filepaths.append(filepath)

            img.download(filepath, override=False)

            pil_image = Image.open(filepath)
            shape = (pil_image.height, pil_image.width)
            mask = np.zeros((shape[0], shape[1]))

            if annot_per_image[img.id]:
                n_img = n_img + 1
                for g in annot_per_image[img.id]:
                    poly_points = []
                    for x, y in g.exterior.coords[:]:
                        poly_points.append([x, y])
                    poly_points = np.array(poly_points)
                    poly_points = np.int32(poly_points)
                    cv2.fillPoly(mask, [poly_points], color=(255))
                cv2.imwrite(os.path.join(mask_path, name), np.flipud(mask))
            else:
                n_empty = n_empty + 1
        print("Number of images downloaded: {}".format(n_img))
        print("Number of images without the ROI: {}".format(n_empty))
    return
