import sys, os
sys.path.insert(0, "/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp")

import torch
import json

from transformers import DetrForObjectDetection
# from models.ddetrhod_dev import DeformableDetrForObjectDetection
from models.ddetrhod import DeformableDetrForObjectDetection       # Debugging model

from datasets import load_dataset



def load_model(mdl:str, cp:str = None, nr_queries:int = 20, num_cls:int = 2, task:str = 'hand', *,
                is_two_stage:bool = False, from_pretrained:bool = False, is_train:bool = False):
    
    ''' Load model based on given args '''

    model = None    # initialize empty model 
    categories = load_dataset('/home/swdev/contactEst/InteractionDetectorDDETR/hoi-comp/data/100DOH', trust_remote_code=True)["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    id2label.update({4:"noon detection"})
    label2id = {v: k for k, v in id2label.items()}


    if cp:
        # load model for single stage Deformable DETR
        model, cp, cfg = _model_resize(mdl=mdl, 
                                       cp=cp, 
                                       nr_queries=nr_queries,
                                       num_cls = num_cls, 
                                       is_train=is_train)
    elif mdl == 'detr' and not cp:
        cp = 'detr-resnet-50_finetuned_hand/checkpoint-18400'
    elif mdl == 'ddetr' and not cp and not is_two_stage:
        # cp = "checkpoints/ddetr-custom-preprocessor-res50/checkpoint-199800"
        cp = "checkpoints/ddetr-custom-preprocessor-res50-(cls+1)-few-queries/checkpoint-33200"
    elif mdl == 'ddetr' and is_two_stage:
        # Train from checkpoints
        if from_pretrained:                
            cp = "checkpoints/ddetr-custom-preprocessor-res50-(cls+1)-2-stage-20-queries/checkpoint-6600"
            model = DeformableDetrForObjectDetection.from_pretrained(cp, 
                                                                    label2id = label2id,
                                                                    id2label = id2label,
                                                                ignore_mismatched_sizes = True)
        else:
            from transformers import DeformableDetrConfig
            cfg = DeformableDetrConfig()
            cfg.num_queries = nr_queries
            cfg.label2id = label2id
            cfg.id2label = id2label
            cfg.with_box_refine = True
            cfg.two_stage = True
            cfg.torch_dtype = torch.float64

            model = DeformableDetrForObjectDetection(cfg).to('cuda')
    else:
        raise NotImplementedError('The choosen model not implemented yet!')

    if model is None:
        # load model for single stage Deformable DETR
        model, cp, cfg = _model_resize(mdl=mdl, 
                                       cp=cp, 
                                       nr_queries=nr_queries,
                                       num_cls = num_cls, 
                                       is_train=is_train)
    
    # specify the model task before loading model
    model.specify_model_task(task=task)

    return model, cp, cfg


def _model_resize(mdl, cp, nr_queries:int=20, num_cls:int=4
                  ,*,
                  is_train:bool=True):
    cfg_js = os.path.join(cp, 'config.json')
    with open(cfg_js, 'r') as f:
        cfg = json.load(f)
    if is_train:
        if cfg['num_queries'] != nr_queries:
            cfg['num_queries'] = nr_queries
            qEmbedding = torch.nn.Embedding(nr_queries, 2 * cfg['d_model']) if mdl == 'ddetr' else torch.nn.Embedding(nr_queries, cfg['d_model'])
            model = DeformableDetrForObjectDetection.from_pretrained(cp,
                                                                    ignore_mismatched_sizes=True)
            model.base_model.query_position_embeddings = qEmbedding
            model.config.num_queries = nr_queries
        else:
            model = DeformableDetrForObjectDetection.from_pretrained(cp,
                                                                    ignore_mismatched_sizes=True)
        # check if the number of the output features from the class embedding layer is equal to num_cls + 1(non detection)
        orig_cls_embed = model.class_embed
        nr_orig_cls_embed_layer = len(orig_cls_embed)
        orig_cls_embed_layer = orig_cls_embed[0]
        out_features = orig_cls_embed_layer.out_features
        # if not, use fine-tuned weights to initialize new embeddings
        if out_features != num_cls+1:
            custom_cls_embed = []
            for l in range(nr_orig_cls_embed_layer):
                curr_orig_embed_layer = orig_cls_embed[l]
                out_features, in_features = curr_orig_embed_layer.weight.data.shape
                curr_custom_embed_layer = torch.nn.Linear(in_features = in_features, out_features = num_cls + 1)
                if out_features < num_cls + 1:
                    curr_custom_embed_layer.weight.data[:out_features, :] = curr_orig_embed_layer.weight.data
                else:
                    curr_custom_embed_layer.weight.data = curr_orig_embed_layer.weight.data[:num_cls+1, :]
                custom_cls_embed.append(curr_custom_embed_layer)
            custom_cls_embed = torch.nn.ModuleList(custom_cls_embed)
            model.class_embed = custom_cls_embed
            model.config.num_labels = num_cls + 1
            cfg['id2label'].update({"4": "non detection"})
            cfg['label2id'].update({"non detection": 4})
        with open(cfg_js, "w") as f:
            json.dump(cfg, f, indent=4)
    elif cfg['num_queries'] != nr_queries and not is_train:
        if mdl == 'detr':
            model = DetrForObjectDetection.from_pretrained(cp,
                                                            ignore_mismatched_sizes=True)
        elif mdl == 'ddetr':
            model = DeformableDetrForObjectDetection.from_pretrained(cp, 
                                                                    ignore_mismatched_sizes=True)
    else:
        if mdl == 'detr':
            model = DetrForObjectDetection.from_pretrained(cp)
        elif mdl == 'ddetr':
            model = DeformableDetrForObjectDetection.from_pretrained(cp)
    return model, cp, cfg