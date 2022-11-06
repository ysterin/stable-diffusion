import cv2
import os
import glob
import numpy as np

# root_dir = "/Users/shukistern/PycharmProjects/stable-diffusion"
root_dir = "../../../data/dreambooth"
images_dir = root_dir + '/gal'
save_dir = root_dir + '/gal_cropped'

os.makedirs(save_dir, exist_ok=True)

images = os.listdir(images_dir)

H, W = 512, 512

for image_file in images:
    image_name = os.path.splitext(image_file)[0]
    image_suffix = os.path.splitext(image_file)[1]
    if image_suffix not in [".jpg"]:
        continue
    image_path = os.path.join(images_dir, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_width, img_height = image.shape[1], image.shape[0]
    min_dim = min(img_width, img_height)
    image = image[:min_dim, :min_dim]

    image = cv2.resize(image, (W, H))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f"{image_name}.png"), image)
