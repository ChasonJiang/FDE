
import os
from shutil import copyfile
import uuid

from tqdm import tqdm


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

if __name__ == '__main__':
    root_dir = "dataset/dataset_v1/val/images"
    bar = tqdm(findAllFile(root_dir))
    for filepath in bar:
        # filename = filepath.split(os.sep)[-1].split(".")
        # filename = filename[0] if len(filename)==1 else ".".join(filename)
        filename = uuid.uuid1()
        # copyfile(filepath,os.path.join(root_dir,f"{filename}_0.png"))
        # copyfile(filepath,os.path.join(root_dir,f"{filename}_1.png"))
        os.rename(filepath,os.path.join(root_dir,f"{filename}.png"))
        # print(filename)
