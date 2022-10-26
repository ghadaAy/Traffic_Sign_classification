from email.mime import image
from random import randint
from sre_constants import OPCODES
import albumentations as A
import numpy as np

def Augmentation_albumentation(N:int, M:int, p:float, mode='all'):
        shift_x = np.linspace(0,150,10)
        shift_y = np.linspace(0,150,10)
        rot     = np.linspace(0,180,10)
        shear   = np.linspace(0,10,10)
        sola    = np.linspace(0,256,10)
        post    = [4,4,5,5,6,6,7,7,8,8]
        cont    = [np.linspace(-0.8,-0.1,10), np.linspace(0.1,2,10)]
        bright  = np.linspace(0.1,0.7,10)
        shar    = np.linspace(0.1,0.9,10)

        Aug =[#0 - geometrical
        A.ShiftScaleRotate(rotate_limit=int(rot[M]), p=p),
        #A.ShiftScaleRotate(shift_limit_y=shift_y[M], rotate_limit=0, shift_limit_x=0, shift_limit=shift_y[M], p=p),
        A.Affine(rotate=rot[M], p=p),
        A.Affine(shear=shear[M], p=p),
        A.InvertImg(p=p),
        #5 - Color Based
        A.Equalize(p=p),
        A.Solarize(threshold=sola[M], p=p),
        A.Posterize(num_bits=post[M], p=p),
        A.RandomBrightnessContrast( p=p),
        A.RandomBrightness(limit=bright[M], p=p),
        A.Sharpen(alpha=shar[M], lightness=shar[M], p=p)]

        if mode == "geo": 
            ops = np.random.choice(Aug[0:5], N)
        elif mode == "color": 
            ops = np.random.choice(Aug[5:], N)
        else:
            ops = np.random.choice(Aug, N)
        transforms = A.Compose(ops)
        return transforms, ops
