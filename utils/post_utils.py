import os, sys
sys.path.insert(0, "/home/swdev/contactEst/InteractionDetectorDDETR/ddetr_dv")

import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from PIL import Image, ImageDraw, ImageFont

from .bbox_utils import bbox_utils


class post_utils:
    def __init__(self, lower_thres:float = 0.5) -> None:
        self.lower_thres = lower_thres
        self.AP_curr = None
        self.AR_curr = None


    @staticmethod
    def generalized_iou(prediction: torch.tensor, reference:torch.tensor):
        """
        Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

        Returns:
            `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(prediction) and M = len(reference)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        if reference.dim() == 1:
            reference.unsqueeze_(0)
        if not (prediction[:, 2:] >= prediction[:, :2]).all():
            raise ValueError(f"prediction must be in [x0, y0, x1, y1] (corner) format, but got {prediction}")
        if not (reference[:, 2:] >= reference[:, :2]).all():
            raise ValueError(f"reference must be in [x0, y0, x1, y1] (corner) format, but got {reference}")
        ious = post_utils._batch_compute_iou(prediction=prediction, reference=reference)
        _, Us = post_utils._compute_intersection_and_union(prediction=prediction, reference=reference)

        top_left = torch.min(prediction[:, None, :2], reference[:, :2])
        bottom_right = torch.max(prediction[:, None, 2:], reference[:, 2:])

        width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
        area = width_height[:, :, 0] * width_height[:, :, 1]

        return ious - (area - Us) / area

    @staticmethod
    def _batch_compute_iou(prediction:torch.tensor, reference:torch.tensor):
        """
        Module carries out the computation of iou between good predictions and the reference having same class
        Input arguments: 
         * prediction [x,y,x,y] -> n x 4 tensor, dtype = torch.float32
           n: number of proposals recognized as the given class  
         * reference  [x,y,x,y] -> m x 4 tensor, dtype = torch.float32
           m: number of reference bboxes of the given class
        Output(s):
         * ious                 -> n x m tensor, dtype = torch.float32
        """

        Is, Us = post_utils._compute_intersection_and_union(prediction=prediction, reference=reference)
        ious = Is / Us

        # x_min_pred, y_min_pred, x_max_pred, y_max_pred = prediction.T
        # ious = torch.zeros([nr_proposal, nr_ref])
        # for idx, ref in enumerate(reference):
        #     refs = ref.repeat([nr_proposal, 1])
        #     x_min_ref, y_min_ref, x_max_ref, y_max_ref = refs.T
        #     # Compute intersection
        #     zero_vec = torch.zeros_like(x_min_ref)
        #     x_intersect = torch.max(zero_vec, torch.min(x_max_pred, x_max_ref) - torch.max(x_min_pred, x_min_ref))
        #     y_intersect = torch.max(zero_vec, torch.min(y_max_pred, y_max_ref) - torch.max(y_min_pred, y_min_ref))
        #     Is = x_intersect * y_intersect
        #     # Compute Union
        #     w_pred = x_max_pred - x_min_pred
        #     h_pred = y_max_pred - y_min_pred
        #     w_ref = x_max_ref - x_min_ref
        #     h_ref = y_max_ref - y_min_ref
        #     Us = w_pred * h_pred + w_ref * h_ref - Is
        #     Is, Us = post_utils._compute_intersection_and_union(prediction=prediction, reference=ref)
        #     iou = Is / Us
        #     ious[:,idx] = iou
        return ious
    
    @staticmethod
    def _compute_intersection_and_union(prediction:torch.tensor, reference:torch.tensor):

        """ 
        Compute union between batched boxes and batched reference 
        Input:
            - prediction: batched predictions
            - reference: reference to compare
        Output:
            - Is: Intersections
            - Us: Unions
        """

        x_min_pred, y_min_pred, x_max_pred, y_max_pred = prediction.T
        nr_proposal = prediction.shape[-2]
        if reference.dim() == 1:
            reference.unsqueeze_(0)
        nr_ref = reference.shape[0]

        Is = torch.zeros([nr_proposal, nr_ref]).to(prediction.device)
        Us = torch.zeros_like(Is).to(prediction.device)
        for idx, ref in enumerate(reference):
            
            # Compute intersection
            refs = ref.repeat([nr_proposal, 1])
            x_min_ref, y_min_ref, x_max_ref, y_max_ref = refs.T
            zero_vec = torch.zeros_like(x_min_ref)
            x_intersect = torch.max(zero_vec, torch.min(x_max_pred, x_max_ref) - torch.max(x_min_pred, x_min_ref))
            y_intersect = torch.max(zero_vec, torch.min(y_max_pred, y_max_ref) - torch.max(y_min_pred, y_min_ref))
            I = x_intersect * y_intersect

            # Compute union
            w_pred = x_max_pred - x_min_pred
            h_pred = y_max_pred - y_min_pred
            w_ref = x_max_ref - x_min_ref
            h_ref = y_max_ref - y_min_ref
            U = w_pred * h_pred + w_ref * h_ref - I
            Is[:,idx] = I
            Us[:,idx] = U
        return Is, Us


    @staticmethod
    def visualization(image, results, references,
                      output_path:str=f'{os.getcwd()}/test/'):
        proposal_boxes = bbox_utils.xywh2xyxy(results.get('boxes')) if results.get('boxes') is not None else []
        image_id = None
        if 'contactstates' not in results:
            proposal_labels = results.get('labels') if results.get('labels') is not None else []
            if references:
                ref_labels = references['class_labels'] if 'class_labels' in references.keys() else references['category']
                ref_labels = [ref_labels] if isinstance(ref_labels, int) else ref_labels
                ref_boxes =  bbox_utils.xywh2xyxy(references['boxes'] if 'boxes' in references.keys() else references['bbox'])
                ref_boxes = ref_boxes.unsqueeze(0) if ref_boxes.dim() == 1 else ref_boxes                
                image_id = references['image_id'].item()
            else:
                ref_labels, ref_boxes = None, None
        else:
            hand_ids = results.get('labels')
            proposal_labels = hand_ids.clone().to(results.get('contactstates').dtype)
            right_hand_ids = results.get('handsides').to(torch.bool)
            left_hand_ids = torch.as_tensor([hand_ids[idx] & (not right_hand_ids[idx]) for idx in range(hand_ids.numel())]).to(torch.bool)
            proposal_labels[left_hand_ids] = 1 + results.get('contactstates')[left_hand_ids]
            proposal_labels[right_hand_ids] = 6 + results.get('contactstates')[right_hand_ids]
            if references:
                ref_labels = references['class_labels'].clone().to(references['contactstates'].dtype)
                handsides = references.get('handsides')
                ref_left_ids = (handsides == 0).to(torch.bool)
                ref_right_ids = (handsides == 1).to(torch.bool)
                ref_labels = torch.zeros_like(ref_labels, dtype=ref_labels.dtype)
                ref_labels[ref_left_ids] = 1 + references.get('contactstates')[ref_left_ids]
                ref_labels[ref_right_ids] = 6 + references.get('contactstates')[ref_right_ids]
                ref_boxes =  bbox_utils.xywh2xyxy(references['boxes'] if 'boxes' in references.keys() else references['bbox'])
                ref_boxes = ref_boxes.unsqueeze(0) if ref_boxes.dim() == 1 else ref_boxes
                image_id = references['image_id'].item()
            else:
                ref_labels, ref_boxes = None, None
                
        
        # ref_boxes =  bbox_utils.xywh2xyxy(references['boxes'] if 'boxes' in references.keys() else references['bbox'])
        # ref_boxes = ref_boxes.unsqueeze(0) if ref_boxes.dim() == 1 else ref_boxes
        # image_id = references['image_id'].item()
        post_utils.save_image_with_bounding_boxes(image=image,
                                                    image_id=image_id,
                                                    output_path=output_path,
                                                    proposal_boxes=proposal_boxes,
                                                    proposal_labels=proposal_labels,
                                                    reference_boxes=ref_boxes,
                                                    reference_labels=ref_labels)

    @staticmethod
    def save_image_with_bounding_boxes(image, image_id, output_path, proposal_boxes, proposal_labels, reference_boxes, reference_labels):
        """
        Draw proposal and reference bounding boxes with labels on an image using Pillow and save the result.

        Parameters:
        - image: PIL Image object.
        - image_id: Identifier for the image.
        - output_path: Path to save the image with bounding boxes.
        - proposal_boxes: List of proposal bounding boxes in the format [x_min, y_min, x_max, y_max].
        - proposal_labels: List of proposal labels corresponding to each bounding box.
        - reference_boxes: List of reference bounding boxes in the format [x_min, y_min, x_max, y_max].
        - reference_labels: List of reference labels corresponding to each bounding box.
        """
        assert isinstance(image, Image.Image), 'The image type is expected to be a PIL Image'

        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes, mode='RGBA')

        fontsize = 21
        fontPath = '/mnt/c/Windows/Fonts/arial.ttf'
        font = ImageFont.truetype(fontPath, fontsize)
        
        # Draw proposal bounding boxes and labels
        for box, label in zip(proposal_boxes, proposal_labels):
            x_min, y_min, x_max, y_max = map(int, box)
            # Determine the color and label text based on the label
            color, label_text = post_utils.get_color_and_text(label)
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
            # draw.rectangle([x_min, y_min, x_max, y_max], fill=(0, 0, 255, 30))
            txt_pos = (x_min, y_min - 25)
            l, t, r, b = draw.textbbox(txt_pos, label_text, font=font)
            draw.rectangle([l-5, t-5, r+5, b+5], fill=color, outline=color)
            draw.text((x_min, y_min - 25), label_text, fill='black', font=font)

        if reference_boxes is not None and reference_labels is not None:
            # Draw reference bounding boxes and labels
            for box, label in zip(reference_boxes, reference_labels):
                x_min, y_min, x_max, y_max = map(int, box)
                # Determine the color and label text based on the label
                color, label_text = post_utils.get_color_and_text(label)
                label_text += '-ref'
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                # draw.rectangle([x_min, y_min, x_max, y_max], fill=(255, 0, 0, 30))
                txt_pos = (x_min, y_max - 25)
                l, t, r, b = draw.textbbox(txt_pos, label_text, font=font)
                draw.rectangle([l-5, t-5, r+5, b+5], fill='white', outline=color)
                draw.text((x_min, y_max - 25), label_text, fill='black', font=font)

        # Save the image with bounding boxes
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if image_id is None:
            import uuid
            image_id = str(uuid.uuid4())  
        image_with_boxes.save(os.path.join(output_path, f"{image_id}.jpg"))


    @staticmethod
    def get_color_and_text(label):
        """
        Determine color and label text based on the given label.

        Parameters:
        - label: Integer label.

        Returns:
        - color: Color string.
        - label_text: Text to display as label.
        """
        color = "yellow"  # Default color for objects
        if label == 0:
            label_text = "O"
        elif label in range(1, 6):  # Left hand
            color = "blue"
            label_text = "L"
            if label == 1:
                label_text += "-N"
            else:
                label_text += "-P"
        elif label in range(6, 11):  # Right hand
            color = "red"
            label_text = "R"
            if label == 6:
                label_text += "-N"
            else:
                label_text += "-P"
        else:
            raise ValueError("Invalid label")

        return color, label_text


def res_remapping(res_path, output_path, criterion):
    with open(res_path, 'r') as f:
        res_data = json.load(f)
    new_res = []
    if criterion == 'handstate' or criterion == 'handside':
        for item in res_data:
            if not item['category_id']:
                continue
            new_prop = item.copy()
            new_prop['category_id'] = new_prop[criterion]
            new_res.append(new_prop)
    elif criterion == 'hand+obj':
        new_res = res_data
    elif not criterion:
        for item in res_data:
            new_prop = item.copy()
            if not new_prop['category_id']:
                new_prop['category_id'] = 0
            else:
                new_prop['category_id'] = 6 + new_prop['handstate'] if new_prop['handside'] else 1 + new_prop['handstate']
            new_res.append(new_prop)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    res_name = f'res_{criterion}.json' if criterion else 'res_all.json'
    full_out_path = os.path.join(output_path, res_name)
    with open(full_out_path, 'w') as f:
        json.dump(new_res, f, indent=4)

    print(full_out_path)
    return full_out_path


def ddetr_res_remapping(res:dict):
    
    """ remap the result annotation for alignment to yolo """
    
    hand_idx = (res['labels'] == 0)
    res['labels'] = hand_idx
    return res


import numpy as np
import matplotlib.pyplot as plt

def plotPR(Evaluator, criterion, mdl): 
    P = Evaluator.eval['precision']
    x = np.arange(0,1.01,0.01)
    markersize = 12
    txt_d = 0.02
    txt_d_y = 0.01
    cs = ['turquoise', 'skyblue', 'mediumseagreen', 'darkcyan', 'slateblue', 'royalblue']

    if criterion == 'handside':
        pr50_0 = P[0,:,0,0,2]
        pr50_1 = P[0,:,1,0,2]
        pr50_m = (pr50_0 + pr50_1) / 2
        AP50_0 = sum(pr50_0) / pr50_0.size
        AP50_1 = sum(pr50_1) / pr50_1.size
        mAP50 = sum(pr50_m) / pr50_m.size 

        max_r_0 = x[np.nonzero(pr50_0)][-1]
        p_max_r_0 = pr50_0[np.nonzero(pr50_0)][-1]
        max_r_1 = x[np.nonzero(pr50_1)][-1]
        p_max_r_1 = pr50_1[np.nonzero(pr50_1)][-1]

        plt.plot(x, P[0,:,0,0,2], color=cs[0],label=f'Left Hand, AP50: {AP50_0:.3f}')
        plt.plot(x, P[0,:,1,0,2], color=cs[1],label=f'Right Hand, AP50: {AP50_1:.3f}')
        # denote max recall point
        plt.plot(max_r_0, p_max_r_0, '*', color=cs[0], markersize=markersize)
        plt.plot(max_r_1, p_max_r_1, '*', color=cs[1], markersize=markersize)
        plt.text(max_r_0 - txt_d, p_max_r_0 - txt_d_y,  f'({max_r_0:.3f}, {p_max_r_0:.3f})', ha='right', fontsize=10)
        plt.text(max_r_1 + txt_d, p_max_r_1 - txt_d_y,  f'({max_r_1:.3f}, {p_max_r_1:.3f})', ha='left', fontsize=10)
        plt.plot(x, pr50_m, color=cs[5], label=f'Mean Precision, mAP50: {mAP50:.3f}', linewidth=5)
        plt.legend()
        plt.title(f"P-R Curve for Hand Side Detection, \nModel: {'YOLOv8-'+mdl if len(mdl)==1 else mdl}, Metric: IoU@0.5")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

    elif criterion == 'handstate':
        pr50_0 = P[0,:,0,0,2]
        pr50_1 = P[0,:,1,0,2]
        pr50_2 = P[0,:,2,0,2]
        pr50_3 = P[0,:,3,0,2]
        pr50_4 = P[0,:,4,0,2]
        AP50_0 = sum(pr50_0) / pr50_0.size
        AP50_1 = sum(pr50_1) / pr50_1.size
        AP50_2 = sum(pr50_2) / pr50_2.size
        AP50_3 = sum(pr50_3) / pr50_3.size
        AP50_4 = sum(pr50_4) / pr50_4.size

        
        max_r_0 = x[np.nonzero(pr50_0)][-1]
        p_max_r_0 = pr50_0[np.nonzero(pr50_0)][-1]
        max_r_1 = x[np.nonzero(pr50_1)][-1]
        p_max_r_1 = pr50_1[np.nonzero(pr50_1)][-1]
        max_r_2 = x[np.nonzero(pr50_2)][-1] if np.sum(np.nonzero(pr50_2)) else 0
        p_max_r_2 = 0 if max_r_2 == 0 else pr50_2[np.nonzero(pr50_2)][-1]
        max_r_3 = x[np.nonzero(pr50_3)][-1]
        p_max_r_3 = pr50_3[np.nonzero(pr50_3)][-1]
        max_r_4 = x[np.nonzero(pr50_4)][-1]
        p_max_r_4 = pr50_4[np.nonzero(pr50_4)][-1]

        if AP50_1 > 0:
            pr50_m = np.sum(P[0,:,:,0,2], axis=1) / 5
            mAP50 = sum(pr50_m) / pr50_m.size 
            plt.plot(x, P[0,:,0,0,2], color=cs[0],label=f'Hand without Contact, AP50: {AP50_0:.3f}')
            plt.plot(x, P[0,:,1,0,2], color=cs[1],label=f'Hand with Self Contact, AP50: {AP50_1:.3f}')
            plt.plot(x, P[0,:,2,0,2], color=cs[2],label=f'Hand Contacting with Other Person, AP50: {AP50_2:.3f}')
            plt.plot(x, P[0,:,3,0,2], color=cs[3],label=f'Hand Contacting with Portable Object(s), AP50: {AP50_3:.3f}')
            plt.plot(x, P[0,:,4,0,2], color=cs[4],label=f'Hand Contacting with Static Object(s), AP50: {AP50_4:.3f}')
            plt.plot(x, pr50_m, color=cs[5], label=f'Mean Precision, mAP50: {mAP50:.3f}',  linewidth=5)
            
            # denote max recall point

            plt.plot(max_r_0, p_max_r_0, '*',  markersize=markersize, color=cs[0])
            plt.plot(max_r_1, p_max_r_1, '*',  markersize=markersize, color=cs[1])
            plt.plot(max_r_2, p_max_r_2, '*',  markersize=markersize, color=cs[2])
            plt.plot(max_r_3, p_max_r_3, '*',  markersize=markersize, color=cs[3])
            plt.plot(max_r_4, p_max_r_4, '*',  markersize=markersize, color=cs[4])

            plt.text(max_r_0+ txt_d, p_max_r_0 - txt_d_y,  f'({max_r_0:.3f}, {p_max_r_0:.3f})', ha='left', fontsize=10)
            plt.text(max_r_1-0.1, p_max_r_1 + 0.02,  f'({max_r_1:.3f}, {p_max_r_1:.3f})', ha='left', fontsize=10)
            plt.text(max_r_2+ txt_d, p_max_r_2 - 0.05,  f'({max_r_2:.3f}, {p_max_r_2:.3f})', ha='left', fontsize=10)
            plt.text(max_r_3+ txt_d, p_max_r_3 - txt_d_y,  f'({max_r_3:.3f}, {p_max_r_3:.3f})', ha='left', fontsize=10)
            plt.text(max_r_4 - 0.1, p_max_r_4 - 0.05,  f'({max_r_4:.3f}, {p_max_r_4:.3f})', ha='left', fontsize=10)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(f"P-R Curve for Hand State Detection, \nModel: {'YOLOv8-'+mdl if len(mdl)==1 else mdl}, Metric: IoU@0.5")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()

        else:
            pr50_m = (pr50_0 + pr50_3) / 2
            mAP50 = sum(pr50_m) / pr50_m.size
            plt.plot(x, P[0,:,0,0,2], color=cs[0],label=f'Hand without Contact, AP50: {AP50_0:.3f}')
            plt.plot(x, P[0,:,3,0,2], color=cs[1],label=f'Hand Contacting with Portable Object(s), AP50: {AP50_3:.3f}')

            # denote max recall point

            plt.plot(max_r_0, p_max_r_0, '*', color=cs[0], markersize=markersize)
            plt.plot(max_r_3, p_max_r_3, '*', color=cs[1], markersize=markersize)

            plt.text(max_r_0+ txt_d, p_max_r_0 - txt_d_y,  f'({max_r_0:.3f}, {p_max_r_0:.3f})', ha='left', fontsize=10)
            plt.text(max_r_3+ txt_d, p_max_r_3 - txt_d_y,  f'({max_r_3:.3f}, {p_max_r_3:.3f})', ha='left', fontsize=10)

            plt.plot(x, pr50_m, color=cs[5], label=f'Mean Precision, mAP50: {mAP50:.3f}',  linewidth=5)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(f"P-R Curve for Hand State Detection,\n Domain Adaptation Evaluation, \nModel: {'YOLOv8-'+mdl if len(mdl)==1 else mdl}, Metric: IoU@0.5")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()


    elif criterion == 'hand+obj':
        pr50_0 = P[0,:,0,0,2]
        pr50_1 = P[0,:,1,0,2]
        AP50_0 = sum(pr50_0) / pr50_0.size
        AP50_1 = sum(pr50_1) / pr50_1.size
        
        max_r_0 = x[np.nonzero(pr50_0)][-1]
        p_max_r_0 = pr50_0[np.nonzero(pr50_0)][-1]
        max_r_1 = x[np.nonzero(pr50_1)][-1]
        p_max_r_1 = pr50_1[np.nonzero(pr50_1)][-1]
        
        if AP50_0 > 0:
            pr50_m = (pr50_0 + pr50_1) / 2
            mAP50 = sum(pr50_m) / pr50_m.size 
            x1 = plt.plot(x, P[0,:,0,0,2],label=f'Target Object, AP50: {AP50_0:.3f}', color=cs[0])
            x2 = plt.plot(x, P[0,:,1,0,2],label=f'Hand, AP50: {AP50_1:.3f}', color=cs[1])
            xm = plt.plot(x, pr50_m, color=cs[5], label=f'Mean Precision, mAP50: {mAP50:.3f}',  linewidth=5)
    
            # denote max recall point

            r0 = plt.plot(max_r_0, p_max_r_0, '*',  markersize=markersize, color=cs[0])
            r1 = plt.plot(max_r_1, p_max_r_1, '*',  markersize=markersize, color=cs[1])
            r0 = plt.text(max_r_0+ txt_d, p_max_r_0 - txt_d_y,  f'({max_r_0:.3f}, {p_max_r_0:.3f})', ha='left', fontsize=10)
            r1 = plt.text(max_r_1+ txt_d, p_max_r_1 - txt_d_y,  f'({max_r_1:.3f}, {p_max_r_1:.3f})', ha='left', fontsize=10)
            plt.legend()
            plt.title(f"P-R Curve for Hand Object Detection, \nModel: {'YOLOv8-'+mdl if len(mdl)==1 else mdl}, Metric: IoU@0.5")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()
        else:
            plt.plot(x, P[0,:,1,0,2],label=f'Hand, AP50: {AP50_1:.3f}',color=cs[0])
    
            # denote max recall point

            plt.plot(max_r_1, p_max_r_1, '*',  markersize=markersize, color=cs[0])
            plt.text(max_r_1+ txt_d, p_max_r_1 - txt_d_y,  f'({max_r_1:.3f}, {p_max_r_1:.3f})', ha='left', fontsize=10)
            plt.legend()
            plt.title(f"P-R Curve for Hand Detection, \n Domain Adaptation Evaluation, \nModel: {'YOLOv8-'+mdl if len(mdl)==1 else mdl}, Metric: IoU@0.5")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()