import os
import json
import torch

class evalCocoResConverter:
    def __init__(self) -> None:
        self.coco_det_format = {
            'image_id': [],
            'category_id': [],
            'bbox': [],
            'score': [],
        }
        self.full_hand_state_dict = {
            'is_hand': [],
            'handstate': [],
            'handside': [],
        }
        self.supported_constraints = ['handstate', 'handside', 'hand+obj', None]
        self.hand_oriented = False
        self.data = []


    def add_ddetr(self, batch:list[dict], image_ids:list):
        nr_imgs = len(image_ids)
        if len(batch) == 0:
            return # skip zero-det image
        for i in range(nr_imgs):
            res = batch[i]
            if res['labels'].numel() == 0:
                continue
            img_id = int(image_ids[i])
            nr_props = res['labels'].numel()
            for i_prop in range(nr_props):
                det_obj = self.coco_det_format.copy()
                det_obj['image_id'] = img_id
                det_obj['category_id'] = res['labels'][i_prop].clone().detach().to('cpu').tolist()
                det_obj['bbox'] = res['boxes'][i_prop].clone().detach().to('cpu').tolist()
                det_obj['score'] = res['scores'][i_prop].clone().detach().to('cpu').tolist()
                if 'contactstates' in res.keys():
                    det_obj['handstate'] = res['contactstates'][i_prop].clone().detach().to('cpu').tolist()
                if 'handsides' in res.keys():
                    det_obj['handside'] = res['handsides'][i_prop].clone().detach().to('cpu').tolist()
                self.data.append(det_obj)


    def add_yolo(self, batch:list[dict], image_ids:list, criterion:str=None):
        nr_imgs = len(image_ids)
        if len(batch) == 0:
            return # skip zero-det image
        for i in range(nr_imgs):
            res = batch[i]
            if res['labels'].numel() == 0:
                continue
            img_id = int(image_ids[i])
            nr_props = res['labels'].numel()
            if self.supported_constraints.index(criterion) in [0, 1]:
                # handstate or handside
                self.hand_oriented = True
            for i_prop in range(nr_props):
                if self.hand_oriented and res['labels'][i_prop].clone().detach().to('cpu').tolist() == 0:
                    # if task only focus on hand related criteria and the current prop is object
                    continue
                det_obj = self.coco_det_format.copy()
                det_obj['image_id'] = img_id
                det_obj['category_id'] = res['labels'][i_prop].clone().detach().to('cpu').tolist()
                det_obj['bbox'] = res['boxes'][i_prop].clone().detach().to('cpu').tolist()
                det_obj['score'] = res['scores'][i_prop].clone().detach().to('cpu').tolist()
                full_hand_state = self.full_hand_state_dict.copy()
                if det_obj['category_id']:
                    # if the proposal is not an object
                    full_hand_state['is_hand'] = 1
                    if det_obj['category_id'] <= 5 and det_obj['category_id'] >= 1:
                        full_hand_state['handside'] = 0
                        full_hand_state['handstate'] = det_obj['category_id'] - 1   # count from 0
                    elif det_obj['category_id'] >= 6 and det_obj['category_id'] <= 10:
                        full_hand_state['handside'] = 1
                        full_hand_state['handstate'] = det_obj['category_id'] - 6
                if criterion and full_hand_state['is_hand']:
                    det_obj = self.constrainted_remapping(det_obj, full_hand_state, criterion)
                self.data.append(det_obj)

            
                



                
    
    def dump_res_file(self, res_path:str=f'{os.getcwd()}/results/', json_name:str='res.json'):
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        fp = os.path.join(res_path, json_name)
        with open(fp, 'w') as f:
            json.dump(self.data, f, indent=4)
        return fp
    
    def constrainted_remapping(self, det_obj:dict, full_hand_state_dict:dict, constraint:str='handstate'):
        
        ''' Constrainted inference results remapping '''

        constraint = constraint.lower()
        if constraint not in self.supported_constraints:
            raise NotImplementedError(f'The given constraint is not supported! Supported constraints: {self.supported_constraints}')
        
        new_det_obj = {k:v for k,v in det_obj.items()}
        
        if self.supported_constraints.index(constraint) == 0:
            # handstate
            new_det_obj['category_id'] = full_hand_state_dict['handstate']
        elif self.supported_constraints.index(constraint) == 1:
            # handside
            new_det_obj['category_id'] = full_hand_state_dict['handside']
        elif self.supported_constraints.index(constraint) == 2:
            # only hand
            new_det_obj['category_id'] = full_hand_state_dict['is_hand']

        return new_det_obj


def yolo2coco(yolo_xywh):

    """ YOLO box to coco box: [x_c, y_c, w, h] -> [x, y, w, h] """

    yolo_xywh = torch.as_tensor(yolo_xywh).to(torch.float)
    yolo_xywh[..., :2] -= yolo_xywh[..., 2:]/2
    yolo_xywh = yolo_xywh.floor()
    return yolo_xywh


def inverse_remapping(result:dict, criterion:str):

    """ YOLO prediction result remap under the given criterion """

    if sum(result['labels'] == 0) > 0:
        raise ValueError('The given results contain object proposals, which do not need to be remapped!')
    l_ids = (result['labels'] >= 1) * (result['labels'] <= 5)
    r_ids = (result['labels'] >= 6) * (result['labels'] <= 10)
    if criterion == 'handstate':
        result['labels'][l_ids] -= 1
        result['labels'][r_ids] -= 6
    elif criterion == 'handside':
        result['labels'][l_ids] = 0
        result['labels'][r_ids] = 1

    return result