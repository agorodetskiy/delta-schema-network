import os
import imageio

INPUT_DIR_NAME = 'state'
OUTPUT_FILE = 'outgif.gif'

FPS = 99
N_IMAGES_MAX = 100

image_paths = (os.path.join(INPUT_DIR_NAME, file_name)
               for file_name in sorted(os.listdir(INPUT_DIR_NAME))[:N_IMAGES_MAX])

images = [imageio.imread(path) for path in image_paths]

imageio.mimwrite(OUTPUT_FILE, images,
                 fps=FPS,
                 loop=0)

print("Done. The output gif is {}.".format(OUTPUT_FILE))
