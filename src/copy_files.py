import os
import shutil
import random

dir = "D:\\Bostan\\SIMD\\SIMD Original\\validation"
outputdir = "D:\\Bostan\\SIMD\\SIMD\\test"

# Amount of random files you'd like to select
amount = 500

for x in range(amount):
    files = [os.path.splitext(file)[0] for file in os.listdir(dir) if
             os.path.isfile(os.path.join(dir, file)) and os.path.splitext(file)[1] == '.jpg']
    print(len(files))
    if len(files) == 0:
        break
    else:
        file = random.choice(files)
        shutil.move(os.path.join(dir, file + ".jpg"), outputdir)
        shutil.move(os.path.join(dir, file + ".txt"), outputdir)