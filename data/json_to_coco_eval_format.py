import os
import json

from copy import deepcopy


class AnnCocoRemap:

    """ A static class consisting of annotation remapping methods for fine-tuning Deformable DETR on different tasks """

    @staticmethod
    def val_formatted_anns_hand(image_id, objects):
        annotations = []
        for i in range(0, len(objects["id"])):
            new_ann = {
                "id": objects["id"][i],
                "category_id": objects["category"][i],
                "iscrowd": 0,
                "image_id": image_id,
                "area": objects["area"][i],
                "bbox": objects["bbox"][i],
            }
            annotations.append(new_ann)

        return annotations

    @staticmethod
    def val_formatted_anns_hand_obj(image_id, objects):
        annotations = []
        for i in range(0, len(objects['id'])):
            new_ann = {
                'id': objects['id'][i],
                "category_id": objects["category"][i],
                "iscrowd": 0,
                "image_id": image_id,
                "area": objects["area"][i],
                "bbox": objects["bbox"][i],
                "contactstate": objects["contactstate"][i],
                "handside": objects["handside"][i]
            }
            annotations.append(new_ann)
        
        return annotations

    @staticmethod
    def val_formatted_anns_hand_only(image_id, objects):
        annotations = []
        for i in range(0, len(objects['id'])):
            if objects['contactstate'][i] < 0:
                continue
            new_ann = {
                'id': objects['id'][i],
                "category_id": objects["category"][i],
                "iscrowd": 0,
                "image_id": image_id,
                "area": objects["area"][i],
                "bbox": objects["bbox"][i],
                "contactstate": objects["contactstate"][i],
                "handside": objects["handside"][i]
            }
            annotations.append(new_ann)
        
        return annotations

    @staticmethod
    def val_formatted_anns_constrainted(image_id, objects, constraint:str):
        annotations = []
        for i in range(0, len(objects['id'])):
            if objects['contactstate'][i] < 0:
                continue
            
            if constraint == 'handstate':
                new_category = 2 + objects["contactstate"][i]
            elif constraint == 'handside':
                new_category = 2 + objects['handside'][i]
            elif constraint == 'handstate+handside':
                new_category = 2 + objects["contactstate"][i] if not objects['handside'][i] else 7 + objects["contactstate"][i] 

            new_ann = {
                'id': objects['id'][i],
                "category_id": new_category,
                "iscrowd": 0,
                "image_id": image_id,
                "area": objects["area"][i],
                "bbox": objects["bbox"][i],
                "contactstate": objects["contactstate"][i],
                "handside": objects["handside"][i]
            }
            annotations.append(new_ann)
        
        return annotations


    @staticmethod
    def val_formatted_anns_constrainted_yolo(image_id, objects, constraint:str):

        feas_constraints = ['handside', 'handstate', 'hand+obj', None]
        if constraint not in feas_constraints:
            raise ValueError("The given constraint is not supported yet!")
        if feas_constraints.index(constraint) in [0,1]:
            # hand state or hand side prediction
            only_hand = True
        else:
            only_hand = False
        annotations = []
        for i in range(0, len(objects['id'])):

            if only_hand:
                if objects['contactstate'][i] < 0:
                    # skip annotations for target object if the task is not hand oriented
                    continue
                elif constraint == 'handstate':
                    new_category = 0 + objects["contactstate"][i]
                elif constraint == 'handside':
                    new_category = 0 + objects['handside'][i]
                
            else:
                if objects['contactstate'][i] < 0:
                    # remap target object from original id to 0
                    new_category = 0
                else:
                    new_category = 1
                    if not constraint:
                        # all detection
                        new_category = 1 + objects["contactstate"][i] if not objects['handside'][i] else 6 + objects["contactstate"][i] 

            
            # elif constraint == 'handstate+handside':

            new_ann = {
                'id': objects['id'][i],
                "category_id": new_category,
                "iscrowd": 0,
                "image_id": image_id,
                "area": objects["area"][i],
                "bbox": objects["bbox"][i],
                "contactstate": objects["contactstate"][i],
                "handside": objects["handside"][i]
            }
            annotations.append(new_ann)
        
        return annotations


    @staticmethod
    @DeprecationWarning
    def save_hand_obj_annotation_file_images(hand_obj, id2label):
        output_json = {}
        path_output_hand_obj = f"{os.getcwd()}/ddetr_dv/data/hand_obj"

        if not os.path.exists(path_output_hand_obj):
            os.makedirs(path_output_hand_obj)

        path_ann = os.path.join(path_output_hand_obj, 'hand_obj_ann.json')
        path_ann_hand_only = os.path.join(path_output_hand_obj, 'hand_obj_ann_hand_only.json')
        detection_cat_json = [{'id':idx, 'name':id2label[idx]} for idx in id2label]
        output_json["images"] = []
        output_json["annotations"] = []
        output_json_hand_only = deepcopy(output_json)
        for example in hand_obj:
            ann = AnnCocoRemap.val_formatted_anns_hand_obj(example["image_id"], example["objects"])
            ann_hand_only = AnnCocoRemap.val_formatted_anns_hand_only(example["image_id"], example["objects"])
            img_info = {
                    "id": example["image_id"],
                    "width": example["image"].width,
                    "height": example["image"].height,
                    "file_name": f"{example['image_id']}.png",
                }
            output_json["images"].append(img_info)
            output_json["annotations"].extend(ann)
            output_json_hand_only['images'].append(img_info)
            output_json_hand_only["annotations"].extend(ann_hand_only)

        output_json["categories"] = detection_cat_json
        output_json_hand_only['categories'] = detection_cat_json

        if os.path.exists(path_ann):
            os.remove(path=path_ann)
        with open(path_ann, "w") as file:
            json.dump(output_json, file, ensure_ascii=False, indent=4)

        if os.path.exists(path_ann_hand_only):
            os.remove(path=path_ann_hand_only)
        with open(path_ann_hand_only, "w") as file:
            json.dump(output_json_hand_only, file, ensure_ascii=False, indent=4)

        file_ls = os.listdir(path_output_hand_obj)
        test_im_ls = [f for f in file_ls if f.endswith('png')]
        if not len(test_im_ls):
            for im, img_id in zip(hand_obj["image"], hand_obj["image_id"]):
                path_img = os.path.join(path_output_hand_obj, f"{img_id}.png")
                im.save(path_img)

        return path_output_hand_obj, path_ann, path_ann_hand_only

    @staticmethod
    @DeprecationWarning
    def save_constrainted_annotation_file_images(hand_obj, id2label, eval_constraint:str):
        
        constraint = eval_constraint.lower()
        supported_constraints = ['handstate', 'handside', 'handstate+handside']
        assert constraint in supported_constraints

        output_json = {}
        path_output_hand_obj = f"{os.getcwd()}/ddetr_dv/data/hand_obj"

        if not os.path.exists(path_output_hand_obj):
            os.makedirs(path_output_hand_obj)

        path_ann = os.path.join(path_output_hand_obj, 'hand_obj_ann.json')
        path_ann_constrainted = os.path.join(path_output_hand_obj, 'hand_obj_ann_constrainted.json')
        detection_cat_json = [{'id':idx, 'name':id2label[idx]} for idx in id2label.keys()]
        output_json["images"] = []
        output_json["annotations"] = []
        output_json_constrainted = deepcopy(output_json)
        for example in hand_obj:
            ann = AnnCocoRemap.val_formatted_anns_hand_obj(example["image_id"], example["objects"])
            ann_constrainted = AnnCocoRemap.val_formatted_anns_constrainted(example["image_id"], example["objects"], constraint)
            img_info = {
                    "id": example["image_id"],
                    "width": example["image"].width,
                    "height": example["image"].height,
                    "file_name": f"{example['image_id']}.png",
                }
            output_json["images"].append(img_info)
            output_json["annotations"].extend(ann)
            output_json_constrainted['images'].append(img_info)
            output_json_constrainted["annotations"].extend(ann_constrainted)

        output_json["categories"] = detection_cat_json
        output_json_constrainted['categories'] = detection_cat_json

        if os.path.exists(path_ann):
            os.remove(path=path_ann)
        with open(path_ann, "w") as file:
            json.dump(output_json, file, ensure_ascii=False, indent=4)

        if os.path.exists(path_ann_constrainted):
            os.remove(path=path_ann_constrainted)
        with open(path_ann_constrainted, "w") as file:
            json.dump(output_json_constrainted, file, ensure_ascii=False, indent=4)

        file_ls = os.listdir(path_output_hand_obj)
        test_im_ls = [f for f in file_ls if f.endswith('png')]
        if not len(test_im_ls):
            for im, img_id in zip(hand_obj["image"], hand_obj["image_id"]):
                path_img = os.path.join(path_output_hand_obj, f"{img_id}.png")
                im.save(path_img)

        return path_output_hand_obj, path_ann, path_ann_constrainted

    @staticmethod
    def save_annotation_file_images(hand_obj, id2label, eval_constraint:str):

        """ 
        Convert annotations to COCO and save to default path 
        *Args:*
        - `hand_obj`: dataset loaded by huggingface's `load_dataset` method
        - `id2label`: `dict` in format {id: label ...}
        - `eval_constraint`: `str`, the target criterion, feasible input: ['handstate', 'handside', 'hand+obj', None]]
        *Outputs:*
        - `path_to_ds`: path to directory where the images are saved
        - `path_ann_constrainted`: path to the annotation file
        """
        
        constraint = eval_constraint.lower() if isinstance(eval_constraint, str) else eval_constraint
        supported_constraints = ['handstate', 'handside', 'hand+obj', None]
        assert constraint in supported_constraints

        output_json = {}
        path_output_hand_obj = "./data/json_format/coco_gt"

        if not os.path.exists(path_output_hand_obj):
            os.makedirs(path_output_hand_obj)

        path_ann_constrainted = os.path.join(path_output_hand_obj, f'gt_{eval_constraint}.json') if eval_constraint else os.path.join(path_output_hand_obj, 'gt_all.json')
        detection_cat_json = [{'id':idx, 'name':id2label[idx]} for idx in id2label.keys()]
        output_json["images"] = []
        output_json["annotations"] = []
        for example in hand_obj:
            ann_constrainted = AnnCocoRemap.val_formatted_anns_constrainted_yolo(example["image_id"], example["objects"], constraint)
            img_info = {
                    "id": example["image_id"],
                    "width": example["image"].width,
                    "height": example["image"].height,
                    "file_name": f"{example['image_id']}.png",
                }
            output_json['images'].append(img_info)
            output_json["annotations"].extend(ann_constrainted)

        output_json['categories'] = detection_cat_json

        if os.path.exists(path_ann_constrainted):
            os.remove(path=path_ann_constrainted)
        with open(path_ann_constrainted, "w") as file:
            json.dump(output_json, file, ensure_ascii=False, indent=4)

        path_output_ims = '/home/swdev/contactEst/InteractionDetectorDDETR/ddetr_dv/data/hoitest'
        if not os.path.exists(path_output_ims):
            os.makedirs(path_output_ims)
        file_ls = os.listdir(path_output_ims)
        test_im_ls = [f for f in file_ls if f.endswith('png')]
        if not len(test_im_ls):
            # for im, img_id in zip(hand_obj["image"], hand_obj["image_id"]):
            #     path_img = os.path.join(path_output_ims, f"{img_id}.png")
            #     im.save(path_img)
            #     del im
            for sample in hand_obj:
                im = sample['image']
                im_id = sample['image_id']
                path_img = os.path.join(path_output_ims, f'{im_id}.png')
                im.save(path_img)
                del im

        return path_output_ims, path_ann_constrainted

    @staticmethod
    @DeprecationWarning
    def save_hand_annotation_file_images(hand, id2label):
        output_json = {}
        path_output_hand = f"{os.getcwd()}/hand/"

        if not os.path.exists(path_output_hand):
            os.makedirs(path_output_hand)

        path_anno = os.path.join(path_output_hand, "hand_ann.json")
        categories_json = [{"supercategory": "hand", "id": id, "name": id2label[id]} for id in id2label]
        output_json["images"] = []
        output_json["annotations"] = []
        for example in hand:
            ann = AnnCocoRemap.val_formatted_anns_hand(example["image_id"], example["objects"])
            output_json["images"].append(
                {
                    "id": example["image_id"],
                    "width": example["image"].width,
                    "height": example["image"].height,
                    "file_name": f"{example['image_id']}.png",
                }
            )
            output_json["annotations"].extend(ann)
        output_json["categories"] = categories_json

        with open(path_anno, "w") as file:
            json.dump(output_json, file, ensure_ascii=False, indent=4)

        for im, img_id in zip(hand["image"], hand["image_id"]):
            path_img = os.path.join(path_output_hand, f"{img_id}.png")
            im.save(path_img)

        return path_output_hand, path_anno