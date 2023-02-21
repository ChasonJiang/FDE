
import os

from shutil import copyfile,copy
def dump_images(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.png'):
                fullname = os.path.join(root, f)
                yield fullname




if __name__ == '__main__':
    images_dir ="D:\\Downloads\\超级极品高价商业mod全集\\AST\\unzip"
    save_dir = "dataset/new_images/3"

    for item in dump_images(images_dir):
        copy(item, save_dir)
