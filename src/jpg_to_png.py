from PIL import Image
import os

directory = r'D:\Bostan\SIMD\SIMD_NEW\blurred_downsampled_test\4x'
destination = r'D:\Bostan\SIMD\SIMD_NEW\Dataset_for_testing_blurred\SIMD\SIMD_train_LR_bicubic\X4'

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        im = Image.open(os.path.join(directory, filename))
        name=os.path.splitext(filename)[0] + '.png'
        rgb_im = im.convert('RGB')
        print(os.path.join(destination, name))
        rgb_im.save(os.path.join(destination, name))
        #print(os.path.join(directory, filename))
        continue
    else:
        continue