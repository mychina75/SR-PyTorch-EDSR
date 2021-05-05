import os
import shutil
import random

dir = r"D:/Bostan/SIMD/SIMD_NEW/validation"
targetdir = r"D:/Bostan/SIMD/SIMD_NEW/Dataset_for_training/SIMD/SIMD_train_HR"

files = [os.path.splitext(file)[0] for file in os.listdir(dir) if
             os.path.isfile(os.path.join(dir, file)) and os.path.splitext(file)[1] == '.jpg']

print(len(files))

for file in files:
    print(file)
    os.remove(os.path.join(targetdir, "img" + str(file) + ".png"))