import pathlib
for path in pathlib.Path(r"D:\Bostan\COWC\cowc_mini\COWC\COWC_train_LR_bicubic\X4").iterdir():
  if path.is_file():
    old_name = path.stem

    old_extension = path.suffix

    directory = path.parent

    new_name = old_name+ "x4" + old_extension

    path.rename(pathlib.Path(directory, new_name))