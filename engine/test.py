from transformers import DeformableDetrConfig
from datasets.dataset_dict import DatasetDict
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import shutil
import time
import os
import torch
import json

from models.ddetrhod import DeformableDetrForObjectDetection
from data.ds_ddetr import CocoDetection
from data.json_to_coco_eval_format import AnnCocoRemap
from utils.bbox_utils import bbox_utils
from processing.processing import BaseProcessor
from utils.post_utils import post_utils, res_remapping, ddetr_res_remapping
from data.ds_ddetr import ds4DDETR
from utils.eval_utils import evalCocoResConverter

from data.ID2LABEL import (ID2LABEL_ALL, ID2LABEL_HANDOBJ, ID2LABEL_HANDSIDE, ID2LABEL_HANDSTATE)

GENERATE_JSON = False
NUM_DEMO_IMGS = 100
DESIRED_TASK = ['hand', 'hand_obj']
FEASIBLE_CRITERIA = ['handstate', 'handside', 'hand+obj', None]


def test(test_ds: DatasetDict, 
         target_size,
         model: DeformableDetrForObjectDetection,
         cfg: DeformableDetrConfig, 
         task: str,
         data_collator,
         generate_json:bool,
         is_demo:bool=False,
         class_thres:float=0.25,
         nms_thres:float=0.5,
         eval_constraint:str=None
         ):
    
    ''' Create a test engine for metric computation and visualization '''

    assert task in DESIRED_TASK
    generate_json = GENERATE_JSON if not generate_json else generate_json
    test_ds_ddetr = ds4DDETR(test_ds, data_task=task, target_size=target_size)

    hand_only = False
    if FEASIBLE_CRITERIA.index(eval_constraint) in [0,1]:
        hand_only = True

    if is_demo:
        import random
        # random.seed(0)
        curr_nr_queries = cfg['num_queries']
        output_path = f'{os.getcwd()}/test/{curr_nr_queries}-queries/'
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        test_img_ids = list(range(9983))
        demo_img_ids = random.sample(test_img_ids, NUM_DEMO_IMGS)
        
    if generate_json:
        if eval_constraint == 'handstate':
            id2label = ID2LABEL_HANDSTATE
        elif eval_constraint == 'handside':
            id2label = ID2LABEL_HANDSIDE
        elif eval_constraint == 'hand+obj':
            id2label = ID2LABEL_HANDOBJ
        elif not eval_constraint:
            id2label = ID2LABEL_ALL
        start = time.time()
        path_to_ds, path_to_ds_json = AnnCocoRemap.save_annotation_file_images(test_ds, id2label, eval_constraint)
        print(f'Created annotation file in .json format under the path: {path_to_ds_json}! (t={(time.time()-start):.2f}s)')
    else:
        path_to_ds = f"{os.getcwd()}/ddetr_dv/data/hand_obj"
        path_to_ds_json = f"/home/swdev/contactEst/InteractionDetectorDDETR/coco_gt_json/gt_{eval_constraint if eval_constraint else 'all'}.json"
        
    test_coco = CocoDetection(test_ds_ddetr, path_to_ds, path_to_ds_json)
    testLoader = torch.utils.data.DataLoader(test_coco, batch_size=1, shuffle=False, num_workers=1, collate_fn=data_collator)
    model = model.to('cuda')
    detConverter = evalCocoResConverter()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(testLoader)):
            pixel_values = data['pixel_values'].to(model.device)
            pixel_mask = data['pixel_mask'].to(model.device)
            labels = [
                {k:v for k,v in t.items()} for t in data['labels']
            ]
            labels = [{k:bbox_utils.rel2abs(v, 800, 1333) if k=='boxes' else v for k,v in lab.items()} for lab in labels]
            labels = [{k:bbox_utils.xyxy2xywh(v) if k=='boxes' else v for k,v in lab.items()} for lab in labels]
            outputs = model(pixel_values, pixel_mask)
            results = BaseProcessor.post_process(outputs)
            results = [{k:v[res['scores'] > class_thres] for k,v in res.items()} for res in results]
            # check if x2y2 > x1y1
            results = [BaseProcessor.check_valid_preds(res) for res in results]
            
            # Neglect object detection, only for hand only metric
            if hand_only:
                hand_results = []
                for res in results:
                    hand_props_idx = (res['labels'] == 0)
                    hand_res = {k:v[hand_props_idx, :] if k == 'boxes' else v[hand_props_idx] for k, v in res.items()}
                    # hand_results.append(hand_res)
                    hand_results.append(ddetr_res_remapping(hand_res))
                results = hand_results
            else:
                results = [ddetr_res_remapping(res) for res in results]
                        
            nms_results = BaseProcessor.get_nms_results(results, nms_thres=nms_thres)            
            img_ids = [label['image_id'] for label in labels] 
            detConverter.add_ddetr(nms_results, img_ids)

            # DEMO VISUALIZATION
            curr_demo_img_ids = [ids for ids in img_ids if ids in demo_img_ids]
            if len(curr_demo_img_ids):
                for ids in curr_demo_img_ids:
                    references = labels[img_ids.index(ids)]
                    demo_results = nms_results[img_ids.index(ids)] if len(nms_results) else dict()
                    post_utils.visualization(test_ds[idx]["image"],
                                            results = demo_results, 
                                            references= references,
                                            output_path=output_path)

            del data

    output_path = f'results/ddetrhod{model.config.num_queries}'
    res_filepath = output_path

    res_filename = f'res_{eval_constraint}.json' if eval_constraint else 'res.json'
    fp = detConverter.dump_res_file(res_path=res_filepath, json_name=res_filename)
    fp = res_remapping(fp, output_path, eval_constraint)
    print(f"Annotation {path_to_ds_json.split('/')[-1]} loaded!")
    cocoGt = COCO(path_to_ds_json)
    cocoDt = cocoGt.loadRes(fp)
    print(f"Prediction result {fp.split('/')[-1]} loaded!")
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()

