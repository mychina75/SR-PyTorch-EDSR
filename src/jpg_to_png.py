from PIL import Image
import os

directory = r'D:\Bostan\COWC\cowc_train+val'
destination = r'D:\Bostan\COWC\cowc_mini\COWC\COWC_train_HR'

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