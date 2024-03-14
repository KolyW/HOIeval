import os, shutil
import json
import torch

from torch.utils.data import Dataset
from PIL import Image

from typing import Any, List

from utils.eval_utils import evalCocoResConverter


class DOHForYOLO(Dataset):

    """ This module creates a 100DOH dataset specific for YOLOHODtor """

    def __init__(
        self, 
        ref_json_path:str = None,
        img_path:str = None,
        task:str = 'handside',
    ) -> None:
        
        """ 
        ref_json_path: path to an existing annotation directory 
        img_path: path to imgs of dataset, containing 3 sub directories
        task: feas_task = ['handside', 'handstate']
        """

        super(DOHForYOLO, self).__init__()

        self.feas_subset = ['train', 'val', 'test']
        self.feas_task = [None, 'handside', 'handstate']
        
        # validate the input path
        self.ref_json_path = ref_json_path
        if self.ref_json_path is None:
            self.ref_json_path = '/home/swdev/contactEst/InteractionDetectorDDETR/json_format/'

        if task not in self.feas_task:
            raise ValueError('The given task is not supported!')
        else:
            self.ref_json_path = os.path.join(self.ref_json_path, task) if task is not None else self.ref_json_path

        # get json ann file names
        self.json_ann_ls = os.listdir(self.ref_json_path)
        
        # get img paths for different subsets 
        if img_path is None:
            img_path = "/home/swdev/contactEst/InteractionDetectorDDETR/yolo_dv/data/100DOH/images"

        if set(os.listdir(img_path)) != set(self.feas_subset):
            raise ValueError(f'The given image path should contain 3 sub directories: train, val, test; but recieved: {os.listdir(img_path)}')
        else:
            self.train_img_p = f"{img_path}/train"
            self.test_img_p = f"{img_path}/test"
            self.val_img_p = f"{img_path}/val"



    def __getitem__(self, index) -> Any:
        
        """ Get train/val/test subset. """

        if not isinstance(index, str) or index not in self.feas_subset:
            raise ValueError("The given arg does not belong to the subsets!")
        json_fname = [f for f in self.json_ann_ls if index in f][0]
        json_path = os.path.join(self.ref_json_path, json_fname)
        if index == 'train':
            im_path = self.train_img_p
        elif index == 'test':
            im_path = self.test_img_p
        elif index == 'val':
            im_path = self.val_img_p
        return DOHSubset(json_path=json_path, im_path=im_path)



class DOHSubset(Dataset):

    """ This module create a subset accessing the given subset from DOH """
    
    def __init__(self, json_path:str, im_path:str) -> None:
        super(DOHSubset, self).__init__()
        with open(json_path, 'r') as f:
            anns = json.load(f)

        self.objs = anns.get('annotations')
        self.im_infos = anns['images']
        self.im_path = im_path
        
        # for json file generation
        self.converter = evalCocoResConverter()
        self._json_dict = {
            'images':[],
            'annotations':[],
            'categories':[]
        }
        self._id2label_hand_obj = [
            {"id": 0,"name": "hand"},
            {"id": 1,"name": "targetobject"},
            {"id": 2,"name": "background"}
        ]
        self._id2label_handside = [
            {"id": 0,"name": "left hand"},
            {"id": 1,"name": "right hand"},
        ]
        self._id2label_handstate = [
            {"id":1,"name":"hand with no contact"},
            {"id":2,"name":"hand with self contact"},
            {"id":3,"name":"hand contacting with other person"},
            {"id":4,"name":"hand contacting with a portable object"},
            {"id":5,"name":"hand contacting with a static object"},
        ]
        self._id2label_all = [
            {"id":1,"name":"targetobject"},
            {"id":2,"name":"left hand with no contact"},
            {"id":3,"name":"left hand with self contact"},
            {"id":4,"name":"left hand contacting with other person"},
            {"id":5,"name":"left hand contacting with a portable object"},
            {"id":6,"name":"left hand contacting with a static object"},
            {"id":7,"name":"right hand with no contact"},
            {"id":8,"name":"right hand with self contact"},
            {"id":9,"name":"right hand contacting with other person"},
            {"id":10,"name":"right hand contacting with a portable object"},
            {"id":11,"name":"right hand contacting with a static object"},
        ]


    def __getitem__(self, index) -> Any:

        """ Return a sample in ensembled class form for easy access """

        im_info = self.im_infos[index]
        im_path = os.path.join(self.im_path, im_info.get('file_name'))
        im = Image.open(im_path)
        anns = [item for item in self.objs if item.get('image_id') == im_info.get('id')]
        return HOILabelInstance(im, im_info, anns)
    
    def __len__(self) -> int:
        return len(self.im_infos)
    
    def generate_json(self, save_path:str=None, task:str=None) -> None:

        """ Generate a json annotation file in coco format """

        feas_tasks = [None, 'hand', 'handstate', 'handside', 'all']
        if task not in feas_tasks:
            raise KeyError("The given task is not supported!")

        if save_path is None:
            default_fname = f"{task if task is not None else 'hand_obj'}.json"
            save_path = os.path.join(self.im_path, default_fname)
        if os.path.exists(save_path) and save_path.endswith('.json'):
            os.remove(save_path)

        if task == None or task == 'hand':
            self._json_dict['categories'] = self._id2label_hand_obj
            objs = self.objs
        elif task == 'handstate':
            self._json_dict['categories'] = self._id2label_handstate
            objs = [obj for obj in self.objs if obj['category_id'] == 0]
            objs = [{k: self.converter.single_subcat_remapping(obj[k], obj['contactstate']) if k == 'category_id' else v for k,v in obj.items()} for obj in objs]
        elif task == 'handside':
            self._json_dict['categories'] = self._id2label_handside
            objs = [obj for obj in self.objs if obj['category_id'] == 0]
            objs = [{k: self.converter.single_subcat_remapping(obj[k], obj['handside']) if k == 'category_id' else v for k,v in obj.items()} for obj in objs]
        elif task == 'all':
            self._json_dict['categories'] = self._id2label_all
            objs = [{k: self.converter.multi_subcat_remapping(obj['handside'], obj['contactstate']) if k == 'category_id' and obj[k] == 0 else v for k,v in obj.items()} for obj in objs]

        self._json_dict['images'] = self.im_infos
        self._json_dict['annotations'] = objs

        with open(save_path, 'w+') as f:
            json.dump(self._json_dict, f, indent=4)

        return save_path


        
class HOIOutputPrototype:

    """ Creates a class as the model output prototype for easy assignent """

    def __init__(self) -> None:
        self.image_id = []          # image id    
        self.bbox = []              # bbox [x,y,w,h]
        self.category_id = []       # category id (0,1)
        self.logits = []            # logits for classification
        self.logits_handstate = []  # logits for hand contact state prediction
        self.handstate = []         # contact state category id (0,1,2,3,4)
        self.logits_handside = []   # logits for hand side prediction
        self.handside = []          # handside category id (0 l,1 r)
        
    def get_hands_dict(self):
        # remove all object detections
        hand_ids = (self.category_id[0] == 0)
        return {k:v[0][hand_ids] for k, v in vars(self).items()}




class HOILabelPrototype:

    """ Creates a class for easy access to the reference including image informations """

    def __init__(self) -> None:
        self.im = None          # PIL Image
        self.im_info = None     # width, height, image_id
        self.ann = HOIObjectAnnPrototype()
        

class HOIObjectAnnPrototype:

    """ Creates a class for easy access to the reference """
    
    def __init__(self) -> None:
        self.id = []              # object ids
        self.image_id = []        # image id    
        self.bbox = []            # bbox [x,y,w,h]
        self.objbox = []          # box for the object in contact [x,y,w,h]
        self.category_id = []     # category id (0,1)
        self.area = []            # area
        self.contactstate = []    # contact state
        self.handside = []        # hand side

    def get_hands_dict(self):
        # remove all object detections
        hand_ids = (torch.as_tensor(self.category_id) == 0)
        return {k:torch.as_tensor(v)[hand_ids] for k, v in vars(self).items()}
    


class HOILabelInstance(HOILabelPrototype):

    """ Creates an instance from the label prototype """

    def __init__(self, im:Image.Image, im_info:dict, anns:List[dict]) -> None:
        super(HOILabelInstance, self).__init__()
        self.im = [im]
        self.im_info = im_info
        for a in anns:
            self.ann.id.append(a.get('id'))
            self.ann.image_id.append(a.get('image_id'))
            self.ann.bbox.append(a.get('bbox'))
            self.ann.objbox.append(a.get('objbox'))
            self.ann.category_id.append(a.get('category_id'))
            self.ann.area.append(a.get('area'))
            self.ann.contactstate.append(a.get('contactstate'))
            self.ann.handside.append(a.get('handside'))
            

def sortIms(orig_ds_path:str=None, save_path:str=None):
    sets = ['train', 'test', 'val']
    if orig_ds_path is None:
        orig_ds_path = "/home/swdev/contactEst/InteractionDetectorDDETR/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007"
    if save_path is None:
        save_path = './data/100DOH/images'
    imP = f"{orig_ds_path}/JPEGImages"
    lsP = f'{orig_ds_path}/ImageSets/Main'
    def getTxtContent(file):
        with open(file, 'r') as f:
            imLs = [fn.strip() for fn in f.readlines()]
        return imLs
    imPs = {k:[f'{fn}.jpg' for fn in getTxtContent(os.path.join(lsP, f'{k}.txt'))] for k in sets}
    for ds in sets:
        subset_path = os.path.join(save_path, ds)
        if not os.path.exists(subset_path):
            os.makedirs(subset_path)
        for im in imPs[ds]:
            tgtImfile = os.path.join(subset_path, im)
            origImfile = os.path.join(imP, im)
            shutil.copyfile(origImfile, tgtImfile)


def collate_fn_batched_attr(batch:List[HOILabelInstance]):
    new_batch = HOILabelPrototype()
    im_batch = [d.im[0] for d in batch]
    im_info_batch = {k:[d.im_info[k] for d in batch] for k in batch[0].im_info.keys()}
    anns_batch = [d.ann for d in batch]
    new_batch.im = im_batch
    new_batch.im_info = im_info_batch
    new_batch.ann = anns_batch
    return new_batch