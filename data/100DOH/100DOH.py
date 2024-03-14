import collections
import json
import os
import numpy as np

from typing import Optional
import datasets


_CATEGORIES = ["hand", "targetobject", "background"]
_CONTACTSTATE = ["none", "self", "other", "portable", "stationary"]
_HANDSIDE = ["left", "right"]
_URLS = {
    "first_domain": "/home/ziyu/Documents/hand_object_detector/data/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007",
    "second_domain": "/home/ziyu/Documents/hand_object_detector/data/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007",  
    }


class HAND(datasets.GeneratorBasedBuilder):
    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                "objects": datasets.Sequence(
                    {
                        "id": datasets.Value("int64"),
                        "area": datasets.Value("int64"),
                        "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "objbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "category": datasets.ClassLabel(names=_CATEGORIES),
                        "contactstate": datasets.Value("int64"),
                        "handside": datasets.Value("int64"),
                        "contactleft": datasets.Value("int64"),
                        "contactright": datasets.Value("int64")
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            features=features
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file_path": '/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/data/json_format/train.json',  
                    "image_dir": '/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/data/100DOH/images/train',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file_path": '/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/data/json_format/test.json',
                    'image_dir': "/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/data/100DOH/images/test",          
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file_path": '/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/data/json_format/val.json',
                    "image_dir":"/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/data/100DOH/images/val",
                },
            ),
        ]

    def _generate_examples(self, annotation_file_path, image_dir):
        def process_annot(annot, category_id_to_category):
            return {
                "id": annot["id"],
                "area": annot["area"],
                "bbox": annot["bbox"],
                "objbox": (annot["objbox"]),
                "category": category_id_to_category[annot["category_id"]],
                "contactstate": annot['contactstate'],
                "handside": annot['handside'],
                "contactleft": annot["contactleft"],
                "contactright": annot["contactright"]
            }

        idx = 0
        with open(annotation_file_path, 'r') as f:
            data = json.load(f)

        category_id_to_category = {category["id"]: category["name"] for category in data["categories"]}
        print('category_id_to_category',category_id_to_category)
        image_id_to_annotations = collections.defaultdict(list)
        for annot in data["annotations"]:
            image_id_to_annotations[annot["image_id"]].append(annot)
        # image_id_to_image = {annot["file_name"]: annot for annot in data["images"]}

        for img in data["images"]:
            filename = img["file_name"]
            objects = [process_annot(annot, category_id_to_category) for annot in image_id_to_annotations[img["id"]]]
            path = image_dir + '/' + filename
            yield idx, {
                "image_id": img["id"],
                "image": {"path": path},
                "width": img["width"],
                "height": img["height"],
                "objects": objects,
            }
            idx += 1
