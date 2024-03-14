import sys
sys.path.insert(0, "/home/swdev/contactEst/InteractionDetectorDDETR/ddetr_dv")

import torch
import numpy as np

from torch import Tensor
from torchvision.transforms import transforms
from utils.bbox_utils import bbox_utils
from PIL import Image

from utils.post_utils import post_utils


NMS_IN_SINGLE_CLASS = False


class BaseProcessor:

    """Create a base preprocessor involving some image processing methods"""

    def __init__(self, *args, **kwargs) -> None:
        self.toTensor = transforms.ToTensor()
        self.isResize = None
        self.wResize, self.hResize = None, None
        self.preTf = None
        if len([elem for elem in args if isinstance(elem, transforms.Resize)]):
            self.isResize = True
        # Initialize preprocessing transforms 1st stage
        if not self.preTf: 
            tf = []
            tf = tf + [transforms.RandomHorizontalFlip(p=1.),
                    transforms.RandomAutocontrast(p=1.0)] + [t for t in args if isinstance(t, transforms)]
            self.preTf = transforms.Compose(tf)
        # Initialize resize transforms 2nd stage
        targetSize = kwargs['target_size'] if 'target_size' in kwargs.keys() else None
        if targetSize:
            self.resize2 = transforms.Resize(targetSize, antialias=True)
        else:
            self.resize2 = None

    def __call__(self, batch):
        """Prototype for self.__call__() function"""
        return batch
        
    def _boxResize(self, bbox, rows, cols):
        if self.hResize and self.wResize:
            bbox = bbox_utils.abs2rel(bbox=bbox, row=rows, col=cols)
            bbox = bbox_utils.rel2abs(bbox=bbox, row=self.hResize, col=self.wResize)
        return bbox

    def _imgPreprocessing(self, img):
        # Check if input valid
        isPILImage = isinstance(img, Image.Image)
        isTensor = isinstance(img, Tensor)
        isNp = isinstance(img, np.ndarray)
        assert (isPILImage or isTensor or isNp)     # if is not one of the listed type, assert   

        img = self.toTensor(img)
        out = self.preTf(img).squeeze()
        if self.isResize:
            self.hResize, self.wResize = out.shape[-2:]
        return out 

    @staticmethod
    def _boxResizeForPad(xywh:Tensor, padding_w, padding_h):

        # check valid boxes
        sample_coord = xywh[:, 0]
        valid_bbox = sample_coord != -1

        xywh[valid_bbox, :] = bbox_utils.abs_xywh_2_rel_padded_xywh(xywh[valid_bbox, :], padding_w, padding_h)
        xyxy = bbox_utils.xywh2xyxy(xywh)
        return xyxy

    @staticmethod
    def check_valid_preds(pred):      
        boxes = bbox_utils.rel2abs(pred['boxes'], 800, 1333)
        boxes, valid_box_ids = bbox_utils.check_valid_boxes(boxes)
        # pred = {
        #     'scores':pred['scores'][valid_box_ids],
        #     'labels':pred['labels'][valid_box_ids],
        #     'boxes':boxes
        # }
        pred = {k:v[valid_box_ids] for k,v in pred.items()}
        pred['boxes'] = boxes
        return pred

    @staticmethod
    def collate_fn(batch, *,
                   padding_w:int=1333, padding_h:int=800):
        """ create a prototype collate function. """
        pass

    @staticmethod
    def pad(imgs, *,
            padding_w:int=1333, padding_h:int=800) -> dict:
        pixel_values, pixel_mask = [], []
        if isinstance(imgs, list):
            imgs = torch.cat([img.unsqueeze(0) for img in imgs], dim=0)
        if imgs.dim() == 4: # batched imgs
            for img in imgs:
                h, w = img.shape[-2:]
                w_to_pad = padding_w - w
                h_to_pad = padding_h - h
                assert (w_to_pad >= 0) and (h_to_pad >= 0)
                padded_img = torch.nn.functional.pad(img, [0,w_to_pad,0,h_to_pad])
                mask = torch.zeros_like(padded_img)
                mask[..., :h, :w] = 1.
                pixel_values.append(padded_img)
                pixel_mask.append(mask[0])
            padded_info = {
                'pixel_values': torch.stack(pixel_values, dim=0),
                'pixel_mask':torch.stack(pixel_mask, dim=0).unsqueeze(1)
            }
        elif imgs.dim() == 3: # single img
            h, w = imgs.shape[-2:]
            w_to_pad = padding_w - w
            h_to_pad = padding_h - h
            assert (w_to_pad >= 0) and (h_to_pad >= 0)
            padded_img = torch.nn.functional.pad(imgs, [0,w_to_pad,0,h_to_pad])
            mask = torch.zeros_like(padded_img)
            mask[..., :h, :w] = 1.
            padded_info = {
                'pixel_values': padded_img,
                'pixel_mask':mask[0].unsqueeze(0)
            }
        return padded_info
        
    @staticmethod
    def formatted_anns(image_id, category, area, bbox, width, height):
        anns = {
            "image_id":[image_id],
            "class_labels":category.to(torch.int64),
            "boxes":bbox,   # absolute coordinate yet 
            "area":area,
            "iscrowd":[0],
            "orig_size":Tensor([width, height]).to(torch.int64)
        }
        return anns
    
    @staticmethod
    def post_process(outputs, target_sizes=None):
        """
        Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DeformableDetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """

        out_logits, out_bbox = outputs.logits, outputs.pred_boxes
        valid_props = out_logits.sigmoid().argmax(dim=2) != 4
        out_logits = out_logits[valid_props].unsqueeze(0)
        out_bbox = out_bbox[valid_props].unsqueeze(0)
        device = out_logits.device


        prob = out_logits.sigmoid()
        nr_topk = 100 if prob.numel() > 100 else prob.numel()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), nr_topk, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]
        # boxes = center_to_corners_format(out_bbox)
        boxes = out_bbox
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        labels_hc, labels_hs = None, None
        if 'logits_hc' in outputs.keys():
            out_logits_hc = outputs.logits_hc
            prob_hc = out_logits_hc.sigmoid()
            scores_hc = torch.gather(prob_hc, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 5))
            labels_hc = scores_hc.argmax(-1)
        if 'logits_hs' in outputs.keys():
            out_logits_hs = outputs.logits_hs
            prob_hs = out_logits_hs.sigmoid()
            scores_hs = torch.gather(prob_hs, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))
            labels_hs = scores_hs.argmax(-1)

        # and from relative [0, 1] to absolute [0, height] coordinates
        """If the relative coordinate is neccecary, uncomment the code lines below"""
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(device)
        # boxes = boxes * scale_fct[:, None, :]

        if (labels_hc is None) and (labels_hs is None): 
            results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        else:
            results = [{"scores": s, "labels": l, "boxes": b, "contactstates": l_hc, 'handsides': l_hs} for s, l, b, l_hc, l_hs in zip(scores, labels, boxes, labels_hc, labels_hs)]

        return results

    @staticmethod
    def NMS(boxes:torch.tensor, scores:torch.tensor, nms_ious_thres = 0.7):
        nr_boxes = boxes.shape[-2]
        confirmed_boxes = torch.full([nr_boxes, 1], float('inf'))
        confirmed_boxes[0,0] = 0.
        for ibox in range(nr_boxes):
            if confirmed_boxes[ibox] < ibox:
                continue
            ref = boxes[ibox, ...]
            ious = post_utils.generalized_iou(boxes, ref) # n x 1
            confirmed_boxes[ious >= nms_ious_thres] = ibox
        confirmed_props_indices = confirmed_boxes.squeeze().to(torch.int).tolist()
        if isinstance(confirmed_props_indices, list):
            confirmed_props_indices = list(elem for elem in set(confirmed_props_indices))
        out_boxes = boxes[confirmed_props_indices]
        out_scores = scores[confirmed_props_indices]
        if not out_scores.dim():
            out_scores.unsqueeze_(0)
        if out_boxes.dim() == 1:
            out_boxes.unsqueeze_(0)
        return out_boxes, out_scores, confirmed_props_indices

    @staticmethod
    def get_nms_results(results:dict[Tensor], nms_thres:float=0.7, nms_in_single_class:bool = None):
        nms_results = []
        nms_in_single_class = nms_in_single_class if nms_in_single_class else NMS_IN_SINGLE_CLASS

        for pred in results:
            # check if the current prediction is empty
            if not pred['scores'].numel():
                continue
            if nms_in_single_class:
                
                assert 'handside' not in pred.keys() # TODO: add remapping containing handstate and handside

                # nms within the single class
                confirmed_boxes = []
                confirmed_scores = []
                confirmed_labels = []
                for cls in set(pred['labels'].tolist()):
                    box_ids = pred["labels"] == cls
                    boxes = pred['boxes'][box_ids]
                    scores = pred['scores'][box_ids]
                    single_confirmed_boxes, single_confirmed_scores = BaseProcessor.NMS(boxes, scores, nms_thres)
                    single_confirmed_clses = torch.tensor([cls] * single_confirmed_scores.numel())
                    
                    confirmed_boxes.append(single_confirmed_boxes)
                    confirmed_scores.append(single_confirmed_scores)
                    confirmed_labels.append(single_confirmed_clses)
                if (len(confirmed_scores) > 1):
                    nms_pred = {'scores': torch.cat(confirmed_scores),
                                'boxes': torch.cat(confirmed_boxes),
                                'labels': torch.cat(confirmed_labels)}
                elif len(confirmed_scores) == 1:
                    nms_pred = {'scores': single_confirmed_scores,
                                'boxes': single_confirmed_boxes,
                                'labels': single_confirmed_clses}
                else:
                    nms_pred = {'scores': torch.tensor(confirmed_scores),
                                'boxes': torch.tensor(confirmed_boxes),
                                'labels': torch.tensor(confirmed_labels)}
            else:
                # nms between classes
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']                
                    
                confirmed_boxes_temp, confirmed_scores_temp, confirmed_props_indices = BaseProcessor.NMS(boxes, scores, nms_thres)
                confirmed_labels = labels[confirmed_props_indices]
                confirmed_scores_temp = confirmed_scores_temp if confirmed_scores_temp.dim() else confirmed_scores_temp.unsqueeze(0)
                confirmed_labels = confirmed_labels if confirmed_labels.dim() else confirmed_labels.unsqueeze(0)

                hstate_dict = None
                if ('contactstates' in pred.keys()) and ('handsides' in pred.keys()):
                    labels_hc = pred['contactstates']
                    labels_hs = pred['handsides']
                    confirmed_labels_hc = labels_hc[confirmed_props_indices]
                    confirmed_labels_hs = labels_hs[confirmed_props_indices]
                    confirmed_labels_hc = confirmed_labels_hc if confirmed_labels_hc.dim() else confirmed_labels_hc.unsqueeze(0)
                    confirmed_labels_hs = confirmed_labels_hs if confirmed_labels_hs.dim() else confirmed_labels_hs.unsqueeze(0)
                    hstate_dict = {
                        'contactstates': confirmed_labels_hc,
                        'handsides': confirmed_labels_hs,
                    }
                
                nms_pred = {'scores': confirmed_scores_temp,
                            'boxes': confirmed_boxes_temp,
                            'labels': confirmed_labels,
                            }
                
                if hstate_dict is not None:
                    nms_pred.update(hstate_dict)

            nms_results.append(nms_pred)

        # print(nms_results)
        if len(nms_results):
            nms_results = [
                {k:bbox_utils.xyxy2xywh(v) if k=='boxes' else v for k,v in res.items()}
                            for res in nms_results if res['labels'].numel() > 0
                            ]
        return nms_results
                

class preprocessor_hand_detection(BaseProcessor):
    
    """Create a class for hand detection dataset preparation by delegating from BaseProcessor"""
    
    def __init__(self, *args, **kwargs) -> None:
        super(preprocessor_hand_detection, self).__init__(*args, **kwargs)

    def __call__(self, batch):
        """Preprocessing for hand detection dataset"""
        batch = super().__call__(batch)
        if isinstance(batch, dict):
            # Preprocessing first stage
            if 'image_id' in batch.keys():
                if isinstance(batch['image_id'], int):
                    new_batch = self._single_re_anns(data=batch)
                elif isinstance(batch['image_id'], list):
                    new_batch = self._multiple_re_anns(data=batch)
            # Preprocessing second stage
            elif ('pixel_values' in batch.keys()) and ('pixel_mask' in batch.keys()):
                if self.resize2 is not None:
                    batch["pixel_values"] = self.resize2(batch["pixel_values"])
                    batch['pixel_mask'] = self.resize2(batch["pixel_mask"]).to(bool).squeeze()
                    new_batch = batch
        return new_batch

    def _multiple_re_anns(self, data):

        """ Redo annotation for hand detection in batch """

        image_id = data["image_id"]
        img = data["image"]
        if not self.isResize: 
            width, height = data["width"], data["height"]
        else:
            width, height = self.wResize, self.hResize
        anns = data['objects']

        pixel_values, labels = [], []


        for data_id in range(len(image_id)):
            curr_id = image_id[data_id]
            curr_img = img[data_id]
            if not self.isResize:
                curr_w = width[data_id]
                curr_h = height[data_id]
            else:
                curr_w = width
                curr_h = height
            curr_ann = anns[data_id]
            curr_ann = {k:Tensor(v) if isinstance(v, list) else v for k,v in curr_ann.items()}
            curr_area = curr_ann['area']
            curr_boxes = curr_ann['bbox']
            curr_categories = curr_ann['category']
            curr_pixel_values = self._imgPreprocessing(img=curr_img)
            if self.isResize:
                curr_boxes = self._boxResize(bbox=curr_boxes, rows=curr_h, cols=curr_w)
            curr_labels = self.formatted_anns(image_id=curr_id,
                                              category=curr_categories,
                                              area=curr_area,
                                              bbox=curr_boxes,
                                              width=curr_w,
                                              height=curr_h)
            pixel_values.append(curr_pixel_values)
            labels.append(curr_labels)

        new_anns = {
            'pixel_values':pixel_values,
            'labels':labels
        }
        return new_anns

    def _single_re_anns(self, data):
        
        """ Redo annotation for hand detection in single image """

        image_id = data["image_id"]
        img = data["image"]
        if not self.isResize: 
            width, height = data["width"], data["height"]
        else:
            width, height = self.wResize, self.hResize
        anns = data["objects"]
        anns = {k:Tensor(v) if isinstance(v, list) else v for k,v in anns.items()}
        area = anns['area']
        boxes = anns['bbox']
        categories = anns['category']
        pixel_values = self._imgPreprocessing(img=img)
        if self.isResize:
            boxes = self._boxResize(bbox=boxes, rows=height, cols=width)
        labels = self.formatted_anns(image_id=image_id,
                                        category=categories,
                                        area=area,
                                        bbox=boxes,
                                        width=width, height=height)
        new_anns = {
                'pixel_values':pixel_values,
                'labels':labels
            }
        return new_anns

    @staticmethod
    def collate_fn(batch, *,
                   padding_w:int=1333, padding_h:int=800):
        
        """ Collate function for hand detection """

        if isinstance(batch, list):

            # collate data in remapping stage

            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
            labels = [item['labels'] for item in batch]
            batch = {}
            batch['pixel_values'] = pixel_values
            batch['pixel_mask'] = pixel_mask
            batch['labels'] = labels
        elif isinstance(batch, dict):

            # collate data in Reshape stage

            encoding = BaseProcessor.pad(batch['pixel_values'], padding_w=padding_w, padding_h=padding_h)        
            batch["pixel_values"] = encoding["pixel_values"]
            batch["pixel_mask"] = encoding["pixel_mask"]
            batch["labels"]['boxes'] = BaseProcessor._boxResizeForPad(batch["labels"]['boxes'], padding_w=padding_w, padding_h=padding_h)
        return batch


class preprocessor_hand_obj_det(BaseProcessor):
    
    ''' Create a class for hand obj detection dataset preparation by delegating from BaseProcessor '''

    def __init__(self, *args, **kwargs) -> None:
        super(preprocessor_hand_obj_det, self).__init__(*args, **kwargs)

    def __call__(self, batch):

        """Preprocessing for hand detection dataset"""
        
        batch = super(preprocessor_hand_obj_det, self).__call__(batch)
        if isinstance(batch, dict):
            # Preprocessing first stage
            if 'image_id' in batch.keys():
                if isinstance(batch['image_id'], int):
                    new_batch = self._single_re_anns(data=batch)
                elif isinstance(batch['image_id'], list):
                    new_batch = self._multiple_re_anns(data=batch)
            # Preprocessing second stage
            elif ('pixel_values' in batch.keys()) and ('pixel_mask' in batch.keys()):
                if self.resize2 is not None:
                    batch["pixel_values"] = self.resize2(batch["pixel_values"])
                    batch['pixel_mask'] = self.resize2(batch["pixel_mask"]).to(bool).squeeze()
                    new_batch = batch
        return new_batch

    def _single_re_anns(self, data):
         
        """ Redo annotation for hand detection in single image """

        image_id = data["image_id"]
        img = data["image"]
        if not self.isResize: 
            width, height = data["width"], data["height"]
        else:
            width, height = self.wResize, self.hResize
        anns = data["objects"]
        anns = {k:Tensor(v) if isinstance(v, list) else v for k,v in anns.items()}
        area = anns['area']
        boxes = anns['bbox']
        objboxes = anns['objbox']
        contactstates = anns['contactstate']
        handsides = anns['handside']
        contactleft = anns['contactleft']
        contactright = anns['contactright']
        categories = anns['category']
        pixel_values = self._imgPreprocessing(img=img)
        if self.isResize:
            boxes = self._boxResize(bbox=boxes, rows=height, cols=width)
        labels = self.formatted_anns(image_id=image_id,
                                        category=categories,
                                        area=area,
                                        bbox=boxes,
                                        width=width, height=height)
        
        # Addition properties to labels for hand object detection
        addition_dict = {
            "objboxes": objboxes,
            "contactstates": contactstates.to(torch.int64),
            "handsides": handsides.to(torch.int64),
            "contactleft": contactleft.to(torch.int64),
            "contactright": contactright.to(torch.int64),
        }
        labels.update(addition_dict)
        new_anns = {
            'pixel_values': pixel_values,
            "labels": labels
        }
        return new_anns

    def _multiple_re_anns(self, data):

        """ Redo annotation for hand obj detection in batch """

        image_id = data['image_id']
        img = data['image']
        if not self.isResize: 
            width, height = data["width"], data["height"]
        else:
            width, height = self.wResize, self.hResize
        anns = data['objects']

        pixel_values, labels = [], []

        for data_id in range(len(image_id)):
            curr_id = image_id[data_id]
            curr_img = img[data_id]
            if not self.isResize:
                curr_w = width[data_id]
                curr_h = height[data_id]
            else:
                curr_w = width
                curr_h = height
            curr_ann = anns[data_id]
            curr_ann = {k:Tensor(v) if isinstance(v, list) else v for k,v in curr_ann.items()}
            curr_area = curr_ann['area']
            curr_boxes = curr_ann['bbox']
            curr_objboxes = anns['objbox']
            curr_contactstates = anns['contactstate']
            curr_handsides = anns['handside']
            curr_contactleft = anns['contactleft']
            curr_contactright = anns['contactright']
            curr_categories = anns['category']
            curr_pixel_values = self._imgPreprocessing(img=curr_img)
            if self.isResize:
                curr_boxes = self._boxResize(bbox=curr_boxes, rows=curr_h, cols=curr_w)
            curr_labels = self.formatted_anns(image_id=curr_id,
                                              category=curr_categories,
                                              area=curr_area,
                                              bbox=curr_boxes,
                                              width=curr_w,
                                              height=curr_h)
            # Addition properties to labels for hand object detection
            addition_dict = {
                "objboxes": curr_objboxes,
                "contactstates": curr_contactstates.to(torch.int64),
                "handsides": curr_handsides.to(torch.int64),
                "contactleft": curr_contactleft.to(torch.int64),
                "contactright": curr_contactright.to(torch.int64),
            }
            curr_labels.update(addition_dict)

            # append current item to list
            pixel_values.append(curr_pixel_values)
            labels.append(curr_labels)
        
        new_anns = {
            'pixel_values': pixel_values,
            "labels": labels
        }
        return new_anns

    def collate_fn(batch, *,
                   padding_w:int=1333, padding_h:int=800):
        
        """ Collate function for hand object detection """

        if isinstance(batch, list):
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
            labels = [item['labels'] for item in batch]
            batch = {}
            batch['pixel_values'] = pixel_values
            batch['pixel_mask'] = pixel_mask
            batch['labels'] = labels
        elif isinstance(batch, dict):
            encoding = BaseProcessor.pad(batch['pixel_values'], padding_w=padding_w, padding_h=padding_h)        
            batch["pixel_values"] = encoding["pixel_values"]
            batch["pixel_mask"] = encoding["pixel_mask"]
            batch["labels"]['boxes'] = BaseProcessor._boxResizeForPad(batch["labels"]['boxes'], padding_w=padding_w, padding_h=padding_h)
            batch["labels"]['objboxes'] = BaseProcessor._boxResizeForPad(batch["labels"]['objboxes'], padding_w=padding_w, padding_h=padding_h)
            
        return batch


