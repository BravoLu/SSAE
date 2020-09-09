import json
import os.path as osp
import os
import re
from glob import glob

from .base import BaseDataset

class Market1501(BaseDataset):
    def load(self):
        trainval_path = osp.join(self.root, 'Market-1501-v15.09.15', 'Market1501_trainval.json')
        if not os.path.exists(trainval_path):
            self.register(osp.join(self.root, 'Market-1501-v15.09.15'), 'bounding_box_train', trainval_path, reset=True)
        gallery_path = osp.join(self.root, 'Market-1501-v15.09.15', 'Market1501_gallery.json')
        if not osp.exists(gallery_path):
            self.register(osp.join(self.root, 'Market-1501-v15.09.15'), 'bounding_box_test', gallery_path, reset=False)
        query_path = osp.join(self.root, 'Market-1501-v15.09.15', 'Market1501_query.json')
        if not osp.exists(query_path):
            self.register(osp.join(self.root, 'Market-1501-v15.09.15'), 'query', query_path, reset=False)
        for dict in self._load_json(osp.join(self.root, 'Market-1501-v15.09.15/Market1501_trainval.json')):
            fpath = dict['filename']
            # fpath = '/home/shaohao/adversarial-attack-reid/tmp/perturbed_train/%s.jpg'%osp.basename(dict['filename']).split('.')[0]
            self.train.append([
                                fpath,
                                dict['pid'],
                                dict['cam'],
                                osp.basename(dict['filename']).split('.')[0]
                            ])

        for dict in self._load_json(osp.join(self.root,  'Market-1501-v15.09.15/Market1501_query.json')):
            fpath = dict['filename']
            # fpath = '/home/shaohao/adversarial-attack-reid/tmp/baseline/%s.png'%osp.basename(dict['filename']).split('.')[0]
            self.query.append([
                                fpath,
                                dict['pid'],
                                dict['cam'],
                                osp.basename(dict['filename']).split('.')[0]
                            ])

        for dict in self._load_json(osp.join(self.root, 'Market-1501-v15.09.15/Market1501_gallery.json')):
            if dict['pid'] == 0:
                continue
            self.gallery.append([
                                dict['filename'],
                                dict['pid'],
                                dict['cam'],
                                osp.basename(dict['filename']).split('.')[0]
                            ])

        self.test = self.gallery + self.query

    def register(self, absdir, subdir, output, pattern = re.compile(r'([-\d]+)_c(\d)'), reset=True):
        fpaths = sorted(glob(os.path.join(absdir, subdir, '*.jpg')))
        items = []
        pid_sets = {}
        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if reset:
                if pid not in pid_sets:
                    pid_sets[pid] = len(pid_sets)
                pid = pid_sets[pid]
            assert 0 <= pid <= 1501
            assert 1 <= cam <= 6
            cam -= 1
            line = {
                'filename': fpath,
                'pid': pid,
                'cam': cam,
            }
            items.append(line)

        with open(output, 'w') as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
