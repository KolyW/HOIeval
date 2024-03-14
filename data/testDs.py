from torchvision.io import read_image
import os
import torch
import json

from torch.utils.data import Dataset, DataLoader
from typing import Any

class evalDS(Dataset):

    """ A small dataset for domain adapatation test """

    def __init__(self) -> None:
        super(evalDS, self).__init__()
        self.im_path = '/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/frames'
        self.ho_gt_path = '/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/anns/gt_hand+obj.json'
        self.hc_gt_path = '/home/swdev/contactEst/InteractionDetectorDDETR/eval/data/anns/gt_handstate.json'
        with open(self.ho_gt_path, 'r') as f:
            self.gts_ho = json.load(f) 
        with open(self.hc_gt_path, 'r') as f:
            self.gts_hc = json.load(f)
        self.im_info = self.gts_ho.get('images')
        self.im_info = {im['id']:os.path.join(self.im_path, im['file_name']) for im in self.im_info}
        self.anns_ho = self.gts_ho.get('annotations')
        self.anns_hc = self.gts_hc.get('annotations')
        
    def __getitem__(self, index) -> Any:
        if isinstance(index, int) and index < 0:
            index = len(self.anns_hc) + index
        ims = read_image(self.im_info.get(index)) if isinstance(index, int) else torch.cat([read_image(self.im_info.get(ids))[None, ...] for ids in index])
        gts_ho = self.anns_ho[index]['category_id'] if isinstance(index, int) else [self.anns_ho[idx]['category_id'] for idx in index]
        gts_hc = self.anns_hc[index]['category_id'] if isinstance(index, int) else [self.anns_hc[idx]['category_id'] for idx in index]
        return (ims, gts_ho, gts_hc, index)

    def __len__(self):
        return len(self.anns_hc)
    

if __name__ == '__main__':
    ds = evalDS()
    print(len(ds))
    loader = DataLoader(ds, batch_size=4)
    for _,_ in enumerate(loader):
        continue
    