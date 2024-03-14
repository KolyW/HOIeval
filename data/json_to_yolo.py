import os, sys
sys.path.insert(0, '/home/swdev/contactEst/InteractionDetectorDDETR/ddetr_dv')

import json
import shutil
import torch

from utils.bbox_utils import bbox_utils


class Json2YoloAnnConverter:
    def __init__(self, save_path:str = f'{os.getcwd()}/yolo_format/labels', ann_path = os.getcwd()):
        self.save_path = save_path
        self.ann_path = ann_path

    def __call__(self, id_to_skip:list=[], constraint:str=None):
        ann_names = ['train.json', 'test.json', 'val.json']
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        for name in ann_names:
            self._dump_ann(ann_name = name, id_to_skip=id_to_skip, constraint=constraint)

    def _dump_ann(self, ann_name:str = 'train.json', id_to_skip:list=[], constraint:str=None):
        json_path = os.path.join(self.ann_path, ann_name)
        save_path = ann_name.split('.')[0]
        out_path = os.path.join(self.save_path, save_path)
        if not os.path.exists(json_path):
            raise ValueError('The given annotation path is invalid!')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(json_path, 'r') as f:
            json_ann = json.load(f)
        imgs = json_ann['images']
        id_info_dicts = self._id2infos(imgs)
        curr_id = 0
        yolo_ann_path = None
        ann_data = ""
        for item in json_ann['annotations']:
            if item['image_id'] in id_to_skip:
                continue
            elif constraint and constraint != 'all' and item['category_id'] == 1:
                # Skip the object gts for hand oriented detection
                continue
            elif curr_id != item['image_id']:
                # if the id changes, dump the saved ann data firstly, then change the current id
                if len(ann_data) and yolo_ann_path and not os.path.exists(yolo_ann_path):
                    with open(yolo_ann_path, 'w') as f:
                        f.write(ann_data.rstrip())
                    ann_data = ""
                curr_id = item['image_id']
                img_w, img_h = id_info_dicts[curr_id]['width'], id_info_dicts[curr_id]['height']
                yolo_ann_path = os.path.join(out_path, f"{id_info_dicts[curr_id]['file_name'].split('.')[0]}"+'.txt')
            curr_box_coco = item['bbox']
            curr_box_yolo = self._coco2yolo(curr_box_coco, img_h, img_w)
            curr_cls = item['category_id']
            if constraint:
                curr_cls = self._constrainted_remapping(item, constraint)
            ann_data += f"{curr_cls} {curr_box_yolo[0]} {curr_box_yolo[1]} {curr_box_yolo[2]} {curr_box_yolo[3]}\n"
                  
    def _coco2yolo(self, coco_box, height, width):
        yolo_box = bbox_utils.abs2rel(torch.tensor(coco_box).to(torch.float), height, width)
        yolo_box[0:2] = yolo_box[0:2]+ yolo_box[2:] / 2
        return yolo_box.tolist()

    def _id2infos(self, ann_images:list[dict]):
        ids = []
        img_infos = []
        for elem in ann_images:
            ids.append(elem['id'])
            img_info = {'width': elem['width'],
                        'height': elem['height'],
                        'file_name':elem['file_name']}
            img_infos.append(img_info)
        id_info_dicts = dict(zip(ids, img_infos))
        return id_info_dicts
    
    def _constrainted_remapping(self, item, constraint:str = None):
        feas_constraints = ['handside', 'handstate', 'all']
        if constraint not in feas_constraints:
            raise ValueError(f'The given constraint is not supported!, the feasible constraints are {feas_constraints}.')
        if constraint == 'handside':
            # handside
            remapped_cls = item['handside'] + 1
        elif constraint == "handstate":
            # handstate
            remapped_cls = item['contactstate'] + 1
        else:
            # all
            if item['category_id'] == 1:
                remapped_cls = 0
            elif item['handside'] == 0:
                remapped_cls = 1 + item['contactstate']
            else:
                remapped_cls = 6 + item['contactstate']
                
        return remapped_cls



if __name__ == "__main__":
    criterion = 'all'
    ann_path = '/home/swdev/contactEst/InteractionDetectorDDETR/json_format'
    save_path = '/home/swdev/contactEst/InteractionDetectorDDETR/yolo_dv/data/100DOH/labels'
    Y_Cter = Json2YoloAnnConverter(ann_path=ann_path, save_path=save_path)
    Y_Cter(constraint=criterion)