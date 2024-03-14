from datasets import load_dataset
from processing.processing import (preprocessor_hand_detection, 
                                    preprocessor_hand_obj_det
                                    )
from data.ds_ddetr import ds4DDETR
from engine.load_model import load_model
from engine.train import train_model
from engine.test import test

import argparse




""" globals for data """
DATAPATH = "./hoi-comp/data/100DOH"
task = 'hand_obj'
IS_TRAIN = False    # set to false if evaluation
IS_DEMO = True
IS_TWO_STAGE = False
generate_json = False
nr_queries = 20
cp = 'hoi-comp/models/checkpoints/DeformableDETR'
# cp = '/home/swdev/contactEst/InteractionDetectorDDETR/checkpoints/ddetrQueries20/checkpoint-200'
# cp = '/home/swdev/contactEst/InteractionDetectorDDETR/checkpoints/ddetrRichQueries/checkpoint-35400'
batch_size = 6
scaling_rate = 2
cls_thres:float=0.1
nms_thres:float=0.5
freeze_dict = {
                'model': False,
                'attn_fusion_layer': False,
                'class_embed': False,
                'ext_hc_embeds': False,
                'ext_hside_embeds': False,
                'bbox_embed': False
            }
eval_constraint = 'hand+obj'   # Feasible constraints: [None (all included), 'handstate', 'handside', 'hand+obj']



def parsearg():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--scaling_rate', type=int, default=2, help='Scale rate to determine the input shape of data')
    parser.add_argument('--is_train', action='store_true', help='Specify to train, false to test')
    parser.add_argument('--is_demo', action='store_false', help='Specify to draw random 100 images during test')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--nr_queries', type=int, default=20, help='desired number of model queries, specify it if a model with different number of queries is to be fit')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate, defualt: 1e-6')
    parser.add_argument('--epochs', type=int, default=10, help='how many epochs for training.')
    parser.add_argument('--decay', type=float, default=1e-4, help='weight decay, defualt: 1e-4')
    parser.add_argument('--save_steps', type=int, default=200, help='how many steps every save')
    parser.add_argument('--logging_steps', type=int, default=50, help='how many steps every log')
    parser.add_argument('--cls_thres', type=float, default=0.2, help='hyperparameter for result filtering, the proposals whose confidence score is lower than this threshold are removed')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='hyperparameter for result filtering, the proposals whose nms score with the proposal having highest legit is higher than this threshold are removed')
    parser.add_argument('--cp', type=int, default=20,help="20 for pre-trained `Deformable DETR with *20* Queries` and 300 for the chackpoint with 300 queries")
    parser.add_argument('--ignore_mismatched_sizes', action='store_true', help='arg for loading model, if true, the model from cp is load regardless the arg `nr_queries`')
    parser.add_argument('--freeze_bb', action='store_true', help='Specify to freeze transformer backbone in training.')
    parser.add_argument('--freeze_heads', action='store_true', help='Specify to freeze detection heads in training.')
    parser.add_argument('--criterion', type=str, help='Description of eval_constraint parameter')

    args = parser.parse_args()
    print(args)
    return args

def ddetr_main(
        cp:str,
        dataset_path:str,
        scaling_rate:int,
        is_train:bool,
        is_demo:bool,
        *,
        epochs:int=10,
        lr:float=1e-6,
        decay:float=1e-4,
        save_steps=200,
        logging_steps=50,
        batch_size:int=6,
        task:str='hand_obj',
        nr_queries:int=300,
        cls_thres:float=0.2,
        nms_thres:float=0.5,
        generate_json:bool=False,
        ignore_mismatched_sizes:bool=True,
        freeze_dict:dict=None,
        eval_constraint:str=None,
):
    
    """
    ### The high-level function for train/test the deformable detr model
    
    #### Input args:
    `cp`: str                     checkpoint to the model, in hugging face format
    `dataset_path`                path to the dataset directory containing {dataset}.py
    `batch_size`: int             batch size for training
    `scalling_rate`               scalling rate for image input resize
    `is_train`                    set to true to train, false to test
    `is_demo`                     set to true to draw random 100 images during test
    
    `task`                        the task that model aims to achieve, currently only "hand_obj" is supported
    `nr_queries`                  desired number of model queries, specify it if a model with different number of queries is to be fit
    `cls_thres`                   hyperparameter for result filtering, the proposals whose confidence score is lower than this threshold are removed
    `nms_thres`                   hyperparameter for result filtering, the proposals whose nms score with the proposal having highest legit is higher than this threshold are removed
    `generate_json`               arg for test phase, if true, generate ground truth json in coco format
    `ignore_mismatched_sizes`     arg for loading model, if true, the model from cp is load regardless the arg `nr_queries`
    `freeze_dict`                 dict to freeze a part of model, for partial training
    `eval_constraint`             Feasible constraints for evaluation: [None (all included), 'handstate', 'handside', 'hand+obj']

    """

    ds = load_dataset(dataset_path, trust_remote_code=True)
    num_cls = 4 if task == 'hand' else 2
    target_size = [1333 // scaling_rate, 800 // scaling_rate]
    
    data_collator = preprocessor_hand_obj_det.collate_fn if task == "hand_obj" else preprocessor_hand_detection.collate_fn
    if ignore_mismatched_sizes:
        from models.ddetrhod import DeformableDetrForObjectDetection
        model = DeformableDetrForObjectDetection.from_pretrained(cp, 
                                                                ignore_mismatched_sizes = True)
        cfg = model.config
    else:
        model, cp, cfg = load_model('ddetr',
                                    cp = cp,
                                    task = task,
                                    nr_queries = nr_queries, 
                                    num_cls = num_cls, 
                                    is_train = is_train)
   
    # Train or Test routine
    if is_train:
        output_dir = f"checkpoints/{cp.split('/')[-2]}"

        # Freeze submodel according freeze dict
        if freeze_dict is not None:
            for n, m in model.named_children():
                if n in freeze_dict.keys():
                    for p in m.parameters():
                        p.requires_grad = not freeze_dict[n]

        # remove 21244, 28484
        indices_to_remove = [21244, 28484]
        train_ds = ds['train'].select(
            (
            i for i in range(len(ds["train"])) 
            if i not in set(indices_to_remove)
            )
        )
        validation_ds = ds['validation']
        train = ds4DDETR(train_ds, data_task=task, target_size = target_size)
        validation = ds4DDETR(validation_ds, data_task=task, target_size = target_size)

        # train model
        train_model(train_ds = train, 
                    validation_ds = validation, 
                    data_collator = data_collator,
                    epochs = epochs,
                    lr=lr,
                    decay=decay,
                    save_steps=save_steps,
                    logging_steps=logging_steps,
                    batch_size = batch_size,
                    model = model,
                    output_dir = output_dir)

    else:
        class_thres = cls_thres
        nms_thres = nms_thres
        generate_json = False
        
        test_ds = ds['test'].select(range(0,9983))
        test(test_ds=test_ds, 
             target_size=target_size,
             model=model,
             cfg=cfg, 
             task=task,
             data_collator=data_collator, 
             is_demo=is_demo,
             class_thres=class_thres,
             nms_thres=nms_thres,
             generate_json=generate_json,
             eval_constraint=eval_constraint)



if __name__ == "__main__":    
    args = parsearg()
    lr = args.lr
    decay = args.decay
    save_steps = args.save_steps
    logging_steps = args.logging_steps
    is_train = args.is_train
    scaling_rate = args.scaling_rate
    is_demo = args.is_demo
    batch_size = args.batch_size
    nr_queries = args.nr_queries
    nms_thres = args.nms_thres
    ignore_mismatched_sizes = args.ignore_mismatched_sizes 
    criterion = args.criterion
    mdl = args.cp
    epoch = args.epochs
    cp = f'{cp}/deform-{mdl}-cp-66600'    
    if args.freeze_heads:
        freeze_dict['bbox_embed'] = True
        freeze_dict['class_embed'] = True
        freeze_dict['ext_hc_embeds'] = True
        freeze_dict['ext_hside_embeds'] = True
    if args.freeze_bb:
        freeze_dict['model'] = True

    ddetr_main(cp,
               dataset_path=DATAPATH,
               batch_size=batch_size,
               scaling_rate=scaling_rate,
               is_train=is_train,
               is_demo=is_demo,
               nr_queries=nr_queries,
               epochs=epoch,
               lr=lr,
               decay=decay,
               save_steps=save_steps,
               logging_steps=logging_steps,
               freeze_dict=freeze_dict,
               task=task,
               generate_json=generate_json,
               cls_thres=cls_thres,
               nms_thres=nms_thres,
               ignore_mismatched_sizes=False,
               eval_constraint=criterion)