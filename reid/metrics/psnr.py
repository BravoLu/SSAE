# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2020-06-12 10:32:34
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2020-06-12 10:33:52
import numpy 
import math 

def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100 
    PIXEL_MAX = 255.0 
    return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))

