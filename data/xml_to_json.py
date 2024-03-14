import xml.etree.ElementTree as ET
import numpy as np
import json
import os, copy


def xmlJsonConverter(datalistPath: str, annPath:str, criterion:str=None) -> dict:

    '''Convert annotation from xml format to json.'''

    # initialize dict with key and empty value pair
    annDict = {'info':{}, # null
               'licenses': [], # null
               'categories': [], # containing categories and class numbers 
               'images':[], # containing file path, id and size
               'annotations': [] # containing annotations
               }
    boxkeys = ['xmin', 'ymin', 'xmax', 'ymax']
    curr_img_id, curr_obj_id = 1, 1

    # set map between ids and categories 
    if criterion is not None:
        if criterion == 'handside':
            categories = [
                {"id":0,"name":"targetobject"},
                {"id":1,"name":"lefthand"},
                {"id":2,"name":"righthand"},
            ]
        elif criterion == 'handstate':
            categories = [
                {"id":0,"name":"targetobject"},
                {"id":1,"name":"hand with no contact"},
                {"id":2,"name":"hand with self contact"},
                {"id":3,"name":"hand contacting with other person"},
                {"id":4,"name":"hand contacting with a portable object"},
                {"id":5,"name":"hand contacting with a static object"},
            ]
        else:
            raise ValueError("The given criterion not supported!")
    else:
        categories = [
                {"id":0,"name":"hand"},
                {"id":1,"name":"targetobject"},
                {"id":2,"name":"background"},
            ]

    # read dataset image path list from txt file
    with open(datalistPath, 'r+') as f:
        TrainDataList = f.read().split('\n')[:-1]
    
    # acquisite annotations 
    ImgInfoList, objdictlist = [], []  
    for filename in TrainDataList:
        
        imgName = f"{filename}.jpg"
        annName = os.path.join(annPath, f"{filename}.xml")

        ImgInfoDict = {"id": curr_img_id, 
                       'width': None,
                       'height': None,
                       "file_name": imgName}

        data = ET.parse(annName)
        root = data.getroot()

        ImgInfoDict["width"] = int(root.find("size").findtext("width"))
        ImgInfoDict["height"] = int(root.find("size").findtext("height"))

        objs = root.findall("object")
        # -1: invalid
        for obj in objs:
            if obj.find("name").text == 'hand':
                objdict = {"name": "hand",
                        "category_id": 0,
                        "id": curr_obj_id,
                        "image_id": curr_img_id,
                        "contactstate": int(obj.find("contactstate").text),
                        "handside": int(obj.find("handside").text),
                        "contactleft": -1,
                        "contactright": -1,
                        "bbox": [int(obj.find('bndbox').find(coord).text) for coord in boxkeys],
                        "objbox": [int(obj.find(f'obj{coord}').text) for coord in boxkeys] if obj.findtext(f'objxmin') != "None" else [-1]*4,
                        "area": [],
                        "iscrowd": int(obj.findtext("difficult"))
                        }
            elif obj.find("name").text == 'targetobject':
                objdict = {"name": "targetobject",
                        "category_id": 1,
                        "id": curr_obj_id,
                        "image_id": curr_img_id,
                        "contactstate": -1,
                        "handside": -1,
                        "contactleft": int(obj.findtext('contactleft')),
                        "contactright": int(obj.findtext('contactright')),
                        "bbox": [int(obj.find('bndbox').find(coord).text) for coord in boxkeys],
                        "objbox": [-1]*4,    # for targetobject no contacting object, use -1 instead
                        "area": [],
                        "iscrowd": int(obj.findtext("difficult"))
                        }
            # compute box area & convert box to xywh
            box = objdict['bbox']
            objdict['bbox'] = _xyxy_2_xywh(box)
            if -1 not in objdict['objbox']:
                # convert box of in-contact object if exists 
                objdict['objbox'] = _xyxy_2_xywh(objdict['objbox'])
            area = _compute_area_xyxy(box)
            objdict['area'] = area

            if criterion is not None:
                objdict = criterion_coupling(objdict, criterion)

            curr_obj_id += 1
            objdictlist.append(objdict)
        
        curr_img_id += 1
        ImgInfoList.append(ImgInfoDict)
    
    # assign values to keys
    annDict["images"] = ImgInfoList
    annDict['categories'] = categories
    annDict["annotations"] = objdictlist
    return annDict

def jsonAnnGtor(datalistPath:str, annPath:str, targetPath:str, criterion:str=None):

    '''dump train, val, and test file to json'''

    if not os.path.exists(targetPath):
        os.mkdir(targetPath)
    setTypes = ['train', 'test', 'val']
    for stype in setTypes:
        setListPath = os.path.join(datalistPath, f'{stype}.txt')
        targetJsonFilePath = os.path.join(targetPath, f'{stype}.json')
        annDict = xmlJsonConverter(datalistPath=setListPath, annPath=annPath, criterion=criterion)
        with open(targetJsonFilePath, 'w') as f:
            json.dump(annDict, f, indent=4)

def _xyxy_2_xywh(bbox:list) -> list:
    '''convert xyxy to xywh'''
    return [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def _compute_area_xyxy(bbox:list) -> int:
    '''compute xyxy bounding box area'''
    return int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

def _compute_area_xywh(bbox:list) -> int:
    '''compute xywh bounding box area'''
    return int((bbox[2] * bbox[3]))

def criterion_coupling(ann: dict, criterion: str=None):
    """combine the class numbers of different criteria and remap to unoccupied class numbers"""
    feas_crit = ['handside', 'handstate', 'handstate+handside']
    if criterion not in feas_crit:
        raise ValueError('Given criterion is not supported!')
    new_ann = copy.deepcopy(ann)
    if criterion == 'handside':
        if new_ann['category_id'] == 0:
            new_ann['category_id'] = 2 + new_ann['handside']
    elif criterion == 'handstate':
        if new_ann['category_id'] == 0:
            new_ann['category_id'] = 2 + new_ann['contactstate']
    return new_ann
