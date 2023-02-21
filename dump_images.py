
import os

from shutil import copyfile,copy
def dump_images(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.png'):
                fullname = os.path.join(root, f)
                yield fullname




if __name__ == '__main__':
    images_dir ="D:\\Downloads\\斗罗大陆\\unzip"
    save_dir = "new_images/7"

    for item in dump_images(images_dir):
        copy(item, save_dir)
