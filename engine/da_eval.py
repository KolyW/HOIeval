import time, random
import torch
from tqdm import tqdm
from utils.eval_utils import evalCocoResConverter, inverse_remapping, yolo2coco
from datasets import load_dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.post_utils import post_utils, plotPR, res_remapping, ddetr_res_remapping
from data.testDs import evalDS

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from processing.processing import BaseProcessor as BP

""" Evaluate the fine tuned model on the test set of 100DOH """
# Select a criterion to evaluate
criteria = ['handstate', 'handside', 'hand+obj', None]
criterion = criteria[0]     # ['handstate', 'handside', 'hand+obj', None]
is_demo = True

def eval_yolo(model, criterion, is_demo:bool=False,*,
              nr_demo_ims:int=100):

    """ Evaluate the model with the given criterion and optionally visualize nr_demo_ims reaults randomly """

    test_ds = load_dataset('/home/swdev/contactEst/InteractionDetectorDDETR/ddetr_dv/data/hoitest', trust_remote_code=True)['test']
    # DEMO hypoparameters
    if is_demo:
        random.seed(0)
        demo_ls = random.sample(list(range(9983)), nr_demo_ims)
        demo_refs = test_ds[demo_ls]

    detConverter = evalCocoResConverter()
    t0 = time.time()
    for idx, data in enumerate(tqdm(test_ds)):
        res = model(data['image'], verbose=False)
        img_id = data['image_id'] 
        scores = res[0].boxes.conf
        boxes = yolo2coco(res[0].boxes.xywh)
        labels = res[0].boxes.cls
        resDict = {'labels': labels,
                'boxes': boxes,
                'scores': scores}
        if criteria.index(criterion) in [0,1]:
            hand_props = (resDict['labels'] != 0)
            resDict = {k: v[hand_props] for k,v in resDict.items()}
        detConverter.add_yolo([resDict], [img_id], criterion=criterion)

        # DEMO VISUALIZATION
        if is_demo:
            output_path = f'/home/swdev/contactEst/InteractionDetectorDDETR/test/yolo-l-{criterion}'
            if img_id-1 in demo_ls:
                ref_id = demo_refs['image_id'].index(img_id)
                references = {k:v[ref_id] for k,v in demo_refs.items()}
                references = references['objects']
                references['image_id'] = torch.as_tensor(data['image_id'])
                references = {k:torch.as_tensor(v) if k != 'image_id' else v for k,v in references.items()}
                if criterion == 'handstate' or criterion == 'handside':
                    hand_refs = (torch.as_tensor(references['category']) == 0)
                    references = {k:v[hand_refs] if k != 'image_id' else v for k,v in references.items()}
                    if criterion == 'handstate':
                        resDict = inverse_remapping(resDict, criterion)
                        references['category'] = references['contactstate']
                    elif criterion == 'handside':
                        resDict = inverse_remapping(resDict, criterion)
                        references['category'] = references['handside']
                elif criterion == 'hand+obj':
                    obj_ids = references['contactstate'] < 0
                    hand_ids = (obj_ids == False)
                    references['category'][obj_ids] = 0
                    references['category'][hand_ids] = 1
                    new_labels = resDict['labels'].clone()
                    hand_ids_res = resDict['labels'] > 0
                    new_labels[hand_ids_res] = 1
                    resDict['labels'] = new_labels
                else:
                    for ref_id in range(references['category'].numel()):
                        if not references['category'][ref_id]:
                            references['category'][ref_id] = 6 + references['contactstate'][ref_id] if references['handside'][ref_id] else 1 + references['contactstate'][ref_id]
                        else:
                            references['category'][ref_id] = 0

                post_utils.visualization(test_ds[idx]["image"],
                                                        results = resDict, 
                                                        references= references,
                                                        output_path=output_path)

        del data

    t1 = time.time()
    print(f'Evaluation time: {t1-t0:.3f}, fps: {9983/(t1-t0):.3f}, device: {model.device}')

    # Dump the detection result to path /results/yolo{model_scale}
    mdl_sz = model.ckpt_path.split('.')[-2][-1]
    res_path = f"/home/swdev/contactEst/InteractionDetectorDDETR/results/yolo{mdl_sz}"
    json_name = f'res_{criterion}.json' if criterion else 'res_all.json'

    fp = detConverter.dump_res_file(res_path=res_path, json_name=json_name)
    return fp


def eval_da_yolo(model, criterion):

    """ Evaluate the domain adaptability of the given YOLO model on the custom dataset """
    feas_crit = ['handstate', 'hand+obj']
    if criterion not in feas_crit:
        raise KeyError("The given criterion is not supported in this custom dataset! ")

    detConverter = evalCocoResConverter()
    test_ds = evalDS()
    loader = DataLoader(test_ds, batch_size=1)
    T_YOLO = transforms.ToPILImage()

    t0 = time.time()
    for _, (ims, _, _, im_ids) in enumerate(tqdm(loader)):
        ims = T_YOLO(ims.squeeze())
        res = model(ims, verbose=False)
        scores = res[0].boxes.conf
        boxes = yolo2coco(res[0].boxes.xywh)
        labels = res[0].boxes.cls
        resDict = {'labels': labels,
                'boxes': boxes,
                'scores': scores}
        if criteria.index(criterion) in [0,1]:
            hand_props = (resDict['labels'] != 0)
            resDict = {k: v[hand_props] for k,v in resDict.items()}
        detConverter.add_yolo([resDict], [im_ids], criterion=criterion)
        del ims

    t1 = time.time()
    print(f'Evaluation time: {t1-t0:.3f}, fps: {20250/(t1-t0):.3f}, device: {model.device}')
    mdl_sz = model.ckpt_path.split('.')[-2][-1]
    res_path = f"/home/swdev/contactEst/InteractionDetectorDDETR/results/yolo{mdl_sz}/da"
    json_name = f'res_{criterion}.json' if criterion else 'res_all.json'
    fp = detConverter.dump_res_file(res_path=res_path, json_name=json_name)
    if criterion:
        gt_pth = f'/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/anns/gt_{criterion}.json'
    else:
        gt_pth = f'/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/anns/gt_all.json'
    print(f'result path: {fp}, gt path: {gt_pth}')
    cocoGt = COCO(gt_pth)
    cocoDt = cocoGt.loadRes(fp)
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    plotPR(E, criterion, mdl_sz)


def eval_plot_COCO(model, fp:str=None, criterion:str=None):

    """ 
    Evaluate the saved detection results with COCOAPI, demonstrate the metric scores, 
    and illustrate the PR curve for metric AP50
    """

    mdl_sz = model.ckpt_path.split('.')[-2][-1]
    if fp is None:
        fp = f"/home/swdev/contactEst/InteractionDetectorDDETR/results/yolo{mdl_sz}/res_{criterion if criterion else 'all'}.json"
        
    if criterion:
        gt_pth = f'/home/swdev/contactEst/InteractionDetectorDDETR/coco_gt_json/gt_{criterion}.json'
    else:
        gt_pth = '/home/swdev/contactEst/InteractionDetectorDDETR/coco_gt_json/gt_all.json'

    print(f'result path: {fp}, gt path: {gt_pth}')
    cocoGt = COCO(gt_pth)
    cocoDt = cocoGt.loadRes(fp)
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()

    plotPR(E, criterion, mdl_sz)


def box_hflip(xyxy):
    newxyxy = xyxy.clone()
    newxyxy[:,2] = 1280 - xyxy[:,0]
    newxyxy[:,0] = 1280 - xyxy[:,2]
    return newxyxy


def eval_da_DDETR(model, criterion):

    """ Evaluate the domain adaptability of ddetr on the custom model """

    if criterion not in ['hand+obj', 'handstate']:
        raise ValueError('The given criterion is not supported on the custom dataset!')
    nr_queries = model.config.num_queries
    class_thres = 0.5
    padw, padh = 1333, 800
    detConverter = evalCocoResConverter()
    test_ds = evalDS()
    loader = DataLoader(test_ds, batch_size=1)
    T_DDETR = transforms.Compose([transforms.Resize([int(padw/2),int(padh/2)], antialias=True)])

    with torch.no_grad():
        t0 = time.time()
        for _, (ims, _, _, im_ids) in enumerate(tqdm(loader)):
            batch = BP.pad(ims, padding_h=padh, padding_w=padw)
            batch = {k:T_DDETR(v) for k,v in batch.items()}
            pixel_value = batch['pixel_values'].cuda() / 255.
            pixel_mask = batch['pixel_mask'].cuda().squeeze(1)
            outputs = model(pixel_value, pixel_mask)
            results = BP.post_process(outputs)
            results = [{k:v[res['scores'] > class_thres] for k,v in res.items()} for res in results]
            # check if x2y2 > x1y1
            results = [BP.check_valid_preds(res) for res in results]
            hand_results = []
            for res in results:
                hand_props_idx = (res['labels'] == 0)
                hand_res = {k:box_hflip(v[hand_props_idx, :]) if k == 'boxes' else v[hand_props_idx] for k, v in res.items()}
                if criterion == 'handstate':
                    hand_res['labels'] = hand_res['contactstates']
                # hand_results.append(hand_res)
                hand_results.append(ddetr_res_remapping(hand_res))
            results = hand_results
            nms_results = BP.get_nms_results(results) 
            nms_results = [{k: 1+res['contactstates'] if k=='labels' else v for k,v in res.items()} for res in nms_results]
            detConverter.add_ddetr(nms_results, [im_ids])

    t1 = time.time()
    print(f'Evaluation time: {t1-t0:.3f}, fps: {20250/(t1-t0):.3f}, device: {model.device}')
    res_path = f"/home/swdev/contactEst/InteractionDetectorDDETR/results/ddetrhod{nr_queries}/da"
    json_name = f'res_{criterion}.json' if criterion else 'res_all.json'
    fp = detConverter.dump_res_file(res_path=res_path, json_name=json_name)

    if criterion:
        gt_path = f'/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/anns/gt_{criterion}.json'
    else:
        print(f'result path: {fp}, gt path: {gt_path}')
    
    fp = res_remapping(fp, res_path, criterion)
    print(f"Annotation {gt_path.split('/')[-1]} loaded!")
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(fp)
    print(f"Prediction result {fp.split('/')[-1]} loaded!")
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    modelnm = f'Deformable DETR {nr_queries} Queries'
    plotPR(E, criterion, modelnm)
    return fp
