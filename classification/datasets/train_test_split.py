# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2020-06-11 16:56:44
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2020-06-11 17:18:07

import random 
import os
import json 

DATA_DIR = '/user/lintao/shaohao/adversarial-attack-by-gan/datas/256_ObjectCategories'

train = []
test = []

for dir in os.listdir(DATA_DIR):
    category = int(dir.split('.')[0])
    imgs = []
    for img in os.listdir(os.path.join(DATA_DIR, dir)):
        if img.endswith('.jpg'):
            imgs.append((os.path.join(DATA_DIR, dir, img), category))

    random.shuffle(imgs)
    num = len(imgs)   
    for t in imgs[:int(num*0.8)]:
        path, cat = t 
        train.append({
            'path':path,
            'category':cat-1
            })
    for t in imgs[int(num*0.8):]:
        path, cat = t 
        test.append({
            'path':path,
            'category':cat-1
            })

print(len(train))
print(len(test))

with open('train.json', 'w') as f:
    json.dump(train, f, indent=4)

with open('test.json', 'w') as f:
    json.dump(test, f, indent=4)
