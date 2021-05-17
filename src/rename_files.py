import pathlib
for path in pathlib.Path(r"D:\Bostan\SIMD\SIMD_NEW\Dataset_for_testing_blurred\SIMD\SIMD_train_LR_bicubic\X4").iterdir():
  if path.is_file():
    old_name = path.stem

    old_extension = path.suffix

    directory = path.parent

    new_name = old_name+ "x4" + old_extension

    path.rename(pathlib.Path(directory, new_name))