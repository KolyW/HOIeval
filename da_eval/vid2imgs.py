import os, json, shutil
import cv2
import random
import numpy as np

from PIL import Image, ImageDraw

""" Convert video to frame data in format of {video_id}_{frame_id}.png  """

def vid2imgs(vid_path:str, output_path):

    """ convert video to frame images """

    data_ls = os.listdir(vid_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    vid_ls = [d for d in data_ls if d.endswith('.mp4') ]
    for vid in vid_ls:
        print(f'Start extracting frames from video: {vid} ...')
        vid_name = vid.split('.')[0]
        v_data = cv2.VideoCapture(os.path.join(vid_path, vid))
        count = 0
        while True:
            success, frame = v_data.read()
            if not success:
                print('Done.')
                break
            save_path_f = os.path.join(output_path, f'{vid_name}_{count}.png')
            cv2.imwrite(save_path_f, frame)
            count += 1



def draw_boxes(json_path, img_path, out_path, num_draw):

    """ draw converted bbox on the original image for debug """

    with open(json_path, 'r') as f:
        gt = json.load(f)
    
    im_infos = gt.get('images')
    im_infos = {im['id']:im['file_name'] for im in im_infos}
    anns = gt.get('annotations')

    nr_ims = len(im_infos)
    im_to_draw = random.sample(range(nr_ims), num_draw)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        shutil.rmtree(out_path)
        os.makedirs(out_path)

    for box in anns:
        if box.get('image_id') not in im_to_draw:
            continue
        curr_f = os.path.join(img_path, im_infos[box.get('image_id')])
        im = Image.open(curr_f)
        im_bbox = im.copy()
        bbox = np.asanyarray(box.get('bbox'))
        bbox[2:] += bbox[:2]
        bbox = list(bbox)
        draw = ImageDraw.Draw(im_bbox, mode='RGBA')
        draw.rectangle(bbox, outline='yellow', fill=(128,128,0,50),  width=5)
        im_bbox.save(os.path.join(out_path, im_infos[box.get('image_id')]))


if __name__ == '__main__':
    json_path = '/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/anns/gt_hand+obj.json'
    img_path = '/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/frames'
    out_path = '/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/debug'
    num_draw = 100
    draw_boxes(json_path, img_path, out_path, num_draw)
        


