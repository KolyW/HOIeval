import torch

from torch import Tensor


class bbox_utils:
    @staticmethod
    def xywh2xyxy(xywh:Tensor):
        # x_min, y_min, w, h -> x_min, y_min, x_max, y_max
        x_min, y_min, w, h = xywh.T
        x_max = x_min + w
        y_max = y_min + h
        xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=0).T
        return xyxy
    
    @staticmethod
    def xyxy2xywh(xyxy:Tensor):
        x_min, y_min, x_max, y_max = xyxy.T
        w = x_max - x_min
        h = y_max - y_min
        xywh = torch.stack([x_min, y_min, w, h], dim=0).T
        return xywh
    
    @staticmethod
    def abs2rel(bbox:Tensor, row, col):
        assert (bbox > 1).any()
        bbox[..., ::2] /= col
        bbox[..., 1::2] /= row
        return bbox.to(torch.float32)
    
    @staticmethod
    def rel2abs(bbox:Tensor, row, col):
        assert (bbox <= 1).all()
        bbox[..., ::2] *= col
        bbox[..., 1::2] *= row
        bbox = bbox.to(dtype=torch.float32)
        return bbox.floor_()
    
    @staticmethod
    def abs_xywh_2_rel_padded_xywh(xywh:Tensor, padding_w, padding_h):
        if not xywh.numel():
            return xywh
        assert (xywh > 1).any()
        xywh[..., ::2] /= padding_w
        xywh[..., 1::2] /= padding_h
        xywh = xywh.to(dtype=torch.float32)
        return xywh

    @staticmethod
    def check_valid_boxes(xyxy:Tensor):
        x_mono = (xyxy[:, 2] > xyxy[:, 0]).to(torch.int)
        y_mono = (xyxy[:, 3] > xyxy[:, 1]).to(torch.int)
        valid_box_ids = (x_mono + y_mono == 2)
        return xyxy[valid_box_ids], valid_box_ids