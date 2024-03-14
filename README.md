## Understanding Hman Object Contact in Robotic Applications

This repository consists of the implementation of the master thesis "Understanding Hman Object Contact in Robotic Applications". Inspired by the convincing detection performance of Faster R-CNN, The master thesis conducted an systematical experiment for assessing large-scale Hand-Object Interaction (HOI) understanding of current State-Of-The-Art (SOTA) Object Detection models, including `YOLOv8` series and `Deformable DETR`, under multiple criteria, including detection performance, real-time capability and domain adaptability. Moreover, the candidate models are fine-tuned on the large-scale HOI dataset, 100 Days Of Hand (`100DOH`), the saved model weights can be directly used in future designs.

### Get code
-------------------

Run the following code in terminal to fetch the implementation:

`git clone <link_to_repo>`

### Requirements
-------------------

Before things get started, creating a specific virtual environment is highly recommanded. 


```
#python3#

cd UnderstandingHOI
deactivate          # deactivate current virtual environment
python3 -m venv .uhoi
. .uhoi/bin/activate
```

#### Install `torch`
The experiment is conducted under `torch=2.1.2` with `cu12.1`. Installation:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Considering the model scale, it is highly recommended to use `cuda` powered version. However, the `cpu` version should work as well.

#### Other requirements

Then, install all requirements:

``pip install -r requirements.txt``

### Dataset
-------------------------

#### `100DOH`

The experiment uses the frame-level subset `100K Frames` of large-scale hand-oriented HOI detection `100DOH` proposed by Dandan Shan in <cite>[Understanding Human Hands in Contact at Internet Scale][frhoi]</cite>. Click <cite>[link][data]</cite> here to jump to the download site.

[frhoi]: https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/file/hands.pdf
[data]: https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html

After downloading, extract the dataset in `pascal_voc` format under same directory as the code. Then open the `data_prep.ipynb` and run it step by step to generate the dataset in required architecture.

When the the data preparation is done, a subfolder named `data/100DOH` should be generated and has following form:

```
| - data
    | -100DOH
        | - images
        | - labels
        | - 100DOH.py
        | - DOH_all.yml
```
where the `100DOH.py` is huggingface dataset script for `Deformable DETR`, and `DOH_all.yml` is configuration file for `YOLO` training.

#### Dataset for domain adaptability evaluation

Except the large-scale dataset, an another dataset is adjusted for the domain adaptability assessment, which can be obtained <cite>[here][mtm]</cite>.

[mtm]: https://drive.google.com/drive/folders/1JVPoCJdb8SNbQFLQAc1zDo0EoM6EukEC

Then, extract the data to path: `da_eval/data/`, and open `da_data_prep.ipynb` under the `da_eval` and run blocks step by step, after conversion the dataset should look like this:

```
|- data_eval
    | - data
        | - anns
        | - frames
        | - mtm_augmented_data      # original dataset
```

### Checkpoints
-------------------------

Checkpoints for pre-trained models can be downloaded via following links: 
| Models          | Checkpoints                                                                             |
|-----------------|-----------------------------------------------------------------------------------------|
| Deformable DETR | https://drive.google.com/drive/folders/1ZjvEZlZm7hameX3jIqsCdiXMFL541xYI?usp=drive_link |
| YOLOv8          | https://drive.google.com/drive/folders/1WIhpn5rJmALsFnATrapLPqWrR81XBC0L?usp=drive_link |

make a new folder `checkpoints` under `./models`

```
mkdir -p ./models/checkpoints
```
And extract downloaded checkpoints into this folder.

### Train
-------------------------

####  Deformable DETR

To train the `deformable DETR` on `100DOH` , run:

```
python3 main.py --is_train --cp=20 --batch_size=4 --lr=1e-6 --decay=1e-4 --logging_steps=200 --save_steps=200
```
This will start 10 epochs training process on `100DOH` trainset with `learning_rate=1e-6` and `weight_decay=1e-4`. The loss will be logged every 200 steps and after logging the model weights is saved. Furthermore, the number of queries can be adjusted by passing `--nr_queries=<nr_queroes>`, and one can also perform partial training by passing `--freeze_bb` to freeze backbone and passing `--freeze_heads` to freeze detection heads.
#### YOLOv8

Check the description in `plots_yolo.ipynb`.

### Detection Performance and Domain Adaptability Evaluation & Plotting
-------------------------
The entire evaluation and result visualization routine are outlined in `plots_ddetr.ipynb` and `plots_yolo.ipynb` corresponding to the employed models. For detailed introduction please go to the mentions jupyter notebooks, amd follow the description.

### Results
-------------------------
The comprehensive model performance evaluation is presented in the following table

|                          | Parameters | FPS        | Hand      | Object    | Hand Side | Hand State | All   | Hand on Custom Dataset | Hand State on Custom Dataset |
|--------------------------|------------|------------|-----------|-----------|-----------|------------|-------|------------------------|------------------------------|
| Faster R-CNN + ResNet101 | 47.3M      | 6.382      | **0.896** | 0.639     | **0.789** | **0.640**  | 0.385 | 0.887                  | 0.360                        |
| YOLOv8-s                 | **11.1M**  | **35.028** | 0.711     | 0.549     | 0.380     | 0.323      | 0.204 | 0.898                  | **0.398**                    |
| YOLOv8-m                 | 25.9M      | 33.652     | 0.769     | 0.649     | 0.416     | 0.378      | 0.232 | **0.903**              | 0.384                        |
| YOLOv8-l                 | 43.6M      | 29.371     | 0.827     | **0.693** | 0.454     | 0.449      | 0.277 | 0.885                  | 0.378                        |
| Deform DETR 20 Queries   | 40.0M      | 15.998     | 0.765     | 0.255     | 0.616     | 0.229      | 0.191 | 0.895                  | 0.308                        |
| Deform DETR 300 Queries  | 40.2M      | 15.125     | 0.782     | 0.243     | 0.646     | 0.238      | 0.203 | 0.849                  | 0.293                        |

