import pathlib
import shutil
from os.path import splitext

i=0
target_dir = r"D:\Bostan\SIMD\SIMD_NEW\Blur_LR_Dataset_for_Detection\validation"
source_dir = r"D:\Bostan\SIMD\SIMD_NEW\Blur_LR_Dataset_for_Detection\train+validation"
for path in pathlib.Path(r"D:\Bostan\SIMD\SIMD_NEW\validation").iterdir():
  if path.is_file():
    name = path.stem
    #new_name = old_name.split("x")[0]

    extension = path.suffix

    directory = path.parent

    if(extension==".jpg"):
        print(name+".png")
        # i = i+1
        # print(i)
        shutil.copy(pathlib.Path(source_dir, name + ".png"), target_dir)