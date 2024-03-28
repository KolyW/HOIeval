import numpy as np
import csv, json
import os
import copy
import cv2


def csv_label_remapping(in_label:str, label_remapping_dict:dict, id2label:dict):

    """ remap csv label from another dataset to the contact detection purpose set """

    curr_key = label_remapping_dict[in_label]
    label2id = {list(dct.values())[1]: list(dct.values())[0] for dct in id2label}
    return label2id[curr_key]


def csvAnn2JsonConverterSingle(csv_path:str, img_path:str, criterion:str, categories:dict, count:int=None): 
      
    """ Convert a single csv annotation file to json format """
    
    curr_csv_file_name = csv_path.split('/')[-1]
    curr_vid_id = curr_csv_file_name.split('_')[0]
    csvfile = open(csv_path, 'r')
    a_table = csv.reader(csvfile)
    feas_crit = ['hand+obj', 'handstate']
    if criterion not in feas_crit:
        raise KeyError('Given criterion is not a feasible one!')

    imgInfosProto = {
        "id": [],
        "width": [],
        "height": [],
        "file_name": []
    }
    label_remapping_dict_hand = {
        'Negative': 'background',
        'Move': 'hand',
        'Release': 'hand',
        'Position': 'hand',
        'Grasp': 'hand',
        'Reach': 'hand',
    }
    label_remapping_dict_handstate = {
        'Negative': 'background',
        'Move': 'hand contacting with a portable object',
        'Release': 'hand contacting with a portable object',
        'Position': 'hand contacting with a portable object',
        'Grasp': 'hand contacting with a portable object',
        'Reach': 'hand with no contact',
    }
    ann_proto = {
        "id": [],
        "category_id": [],
        "iscrowd": 0,
        "image_id": [],
        "area": [],
        "bbox": []
    }

    anns, im_infos = [], []
    width, height = [], []

    if count is None:
        count = 0
    subcount = 0
    for a in a_table:
        
        imgInfos = copy.deepcopy(imgInfosProto)
        ann = copy.deepcopy(ann_proto)

        if len(a[0]) == 0:
            continue
        elif a[-1] == 'Negative':
            curr_frame_path = os.path.join(img_path, f'{curr_vid_id}_{a[0]}.png')
            if os.path.exists(curr_frame_path):
                os.remove(curr_frame_path)
            continue

        idx = int(a[0])
        file_name = f'{curr_vid_id}_{idx}.png'
        if subcount == 0:
            if os.path.exists(os.path.join(img_path, file_name)):
                img = cv2.imread(os.path.join(img_path, file_name))
                height, width = img.shape[:-1]
            else:
                continue

        label = a[-1]

        keypts = [[float(pts.strip('()').split(',')[0])*width, float(pts.strip('()').split(',')[1])*height] for pts in a[1:-1]]    # pts[1]: x, pts[4]: y
        keypts = np.floor(np.asanyarray(keypts))
        x_min, y_min = np.maximum([0,0], np.min(keypts, axis=0).astype(float))
        x_max, y_max = np.minimum([width, height], np.max(keypts, axis=0).astype(float))
        box = [x_min, y_min, x_max-x_min, y_max-y_min]  # [x, y, w, h]
        
        if criterion == 'hand+obj':
            label_remapping_dict = label_remapping_dict_hand
            category_id = csv_label_remapping(label, label_remapping_dict, categories)
        elif criterion == 'handstate':
            label_remapping_dict = label_remapping_dict_handstate
            category_id = csv_label_remapping(label, label_remapping_dict, categories)

        imgInfos["file_name"] = file_name
        imgInfos["height"] = height
        imgInfos['width'] = width
        imgInfos["id"] = count

        ann['id'] = count
        ann['category_id'] = category_id
        ann['is_crowd'] = 0
        ann['image_id'] = count
        ann['area'] = int((x_max - x_min) * (y_max - y_min))
        ann['bbox'] = box

        im_infos.append(imgInfos)
        anns.append(ann)

        count += 1
        subcount += 1

    csvfile.close()

    return im_infos, anns, count


def csv_json_convertor(csv_path:str, output_path:str, criterion:str):

    """ 
    CSV format: mediapipe keypoints, class -> [x, y, z]:float * 18, 1:str
    json format: {
        info: [],
        license: [],
        categories: [], # id to label
        images: [], image infos
        annotations: [], annotations
    }
    """

    # initialize dict with key and empty value pair
    annDict = {'info':{}, # null
               'licenses': [], # null
               'categories': [], # containing categories and class numbers 
               'images':[], # containing file path, id and size
               'annotations': [] # containing annotations
               }

    # set map between ids and categories 
    if criterion is not None:
        if criterion == 'hand+obj':
            categories = [
                {"id":0,"name":"targetobject"},
                {"id":1,"name":"hand"},
                {"id":2,"name":"background"},
            ]
        elif criterion == 'handstate':
            categories = [
                {"id":0,"name":"hand with no contact"},
                {"id":1,"name":"hand with self contact"},
                {"id":2,"name":"hand contacting with other person"},
                {"id":3,"name":"hand contacting with a portable object"},
                {"id":4,"name":"hand contacting with a static object"},
                {"id":5,"name":"background"},
            ]
        else:
            raise ValueError("The given criterion not supported!")
    else:
        raise ValueError("Must give a criterion!")
    
    annDict["categories"] = categories
    csv_ann_ls = [f for f in os.listdir(csv_path) if f.endswith('original.csv')]
    im_infos, anns = [], []

    count = 0
    for f in csv_ann_ls:
        full_path_ann = os.path.join(csv_path, f)
        im_path = output_path
        im_path = im_path.removesuffix(output_path.split('/')[-1])
        im_path = f'{im_path}/frames'
        
        im_info, ann, count = csvAnn2JsonConverterSingle(full_path_ann, im_path, criterion, categories, count)
        im_infos += im_info
        anns += ann

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_path = os.path.join(output_path, 'gt_handstate.json' if criterion == 'handstate' else 'gt_hand.json')
    annDict['images'] = im_infos
    annDict['annotations'] = anns
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'w') as f:
        json.dump(annDict, f, indent=4)



if __name__ == "__main__":
    criterion = 'hand+obj'        # Feasible criterion ['hand+obj', 'handstate']

    csv_path = '/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/da_eval/data/mtm_augmented_data'
    output_path = '/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/da_eval/data/testanns'

    csv_json_convertor(csv_path=csv_path,
                    output_path=output_path,
                    criterion=criterion)
