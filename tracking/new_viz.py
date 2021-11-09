#!/usr/bin/env python3

import numpy as np
import cv2
import glob
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from got10k.datasets import OTB

# change the paths for your system
DATA_PATH = "data/otb2015/"
RES_PATH = "tracking_data/results/OTB2015/ThreeStageTracker_0.06_0.3_0.3_0.1_0.9_7.0"

otb = OTB(root_dir=DATA_PATH, version='tb100', download=False)
for s, (img_files, annos) in enumerate(otb):
    seq_name = otb.seq_names[s]
    print(seq_name)
    try:
        res_file = os.path.join(RES_PATH, seq_name + ".txt")
        with open(res_file) as f:
            ls = [x.strip() for x in f.readlines()]
        assert len(img_files) == len(annos) == len(ls)
        for img_file, ann, res in zip(img_files, annos, ls):
            im = cv2.imread(img_file)

            if res in ("0", "1", "2"):
                if res == "2":
                    # a reset was triggered
                    cv2.waitKey(2000)
                print(res)
            else:
                sp = res.split(",")
                box = np.array([float(x) for x in sp])
                box[2:] += box[:2]
                box = [int(round(x)) for x in box]
                cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 3)

            box = ann.copy()
            box[2:] += box[:2]
            box = [int(round(x)) for x in box]
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            cv2.imshow('Siam R-CNN', im)
            cv2.waitKey(1)
    except Exception as e:
        print(e)