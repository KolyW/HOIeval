"""Extend a subclass CocoDetection"""

import sys
sys.path.insert(0, "/home/swdev/contactEst/InteractionDetectorDDETR/ddetr_dv")

import torch, torchvision
from torch import Tensor
from processing.processing import preprocessor_hand_detection, preprocessor_hand_obj_det


class CocoDetection(torchvision.datasets.CocoDetection):

    """ Create a dataset based on the CocoDetection base class from COCOAPI """

    def __init__(self, ref_ds, img_folder, ann_file):
        super().__init__(img_folder, ann_file)
        self.ref_ds = ref_ds

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        pixel_mask = self.ref_ds[idx]['pixel_mask']
        pixel_values = self.ref_ds[idx]['pixel_values']
        target = {k:torch.as_tensor(v) if not isinstance(v, Tensor) else v 
                  for k,v in self.ref_ds[idx]['labels'].items()}
        target['size'] = torch.tensor([800, 1333], dtype=torch.int64)
        return {"pixel_values": pixel_values, "pixel_mask":pixel_mask, "labels": target}


class ds4DDETR(torch.utils.data.Dataset):

    """ Create a dataset ensembling image, annotations for Deformable DETR """

    def __init__(self, dataset, target_size=[480,480], *,
                 data_task:str='hand'):
        sample = dataset[0]
        self._generate_key_list_for_task(data_task=data_task)
        sample_fields = list(sample['objects'].keys()) if isinstance(sample["objects"], dict) else list(sample['objects'][0].keys())
        if set(sample_fields) != set(self.fields):
            raise ValueError("Prior dataset does not correspond to the given task!")
        self.ds = dataset
        self.task = data_task
        if self.task == "hand":
            self.data_processor = preprocessor_hand_detection(target_size = target_size)
            self.collate_fn = preprocessor_hand_detection.collate_fn
        elif self.task == "hand_obj":
            self.data_processor = preprocessor_hand_obj_det(target_size = target_size)
            self.collate_fn = preprocessor_hand_obj_det.collate_fn
        else:
            raise NotImplementedError("Dataset for given task is not implemented yet! ")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        d = self.data_processor(self.ds[index])
        d = self.collate_fn(d)
        d = self.data_processor(d)
        return d
    
    def _generate_key_list_for_task(self, data_task):
        if data_task == 'hand':
            self.fields = ["id","area","bbox","category"]
        elif data_task == "hand_obj":
            self.fields = ["category", "id",
                           "contactstate", "handside", "contactleft", "contactright",
                           "bbox", "objbox", "area"]
        else:
            raise NotImplementedError("Dataset for the given task not implemented yet!")