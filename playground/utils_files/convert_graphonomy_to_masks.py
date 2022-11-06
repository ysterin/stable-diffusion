import cv2
import os
import glob
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

graphonomy_classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
                      'dress', 'coat', 'socks', 'pants', 'neck', 'scarf', 'skirt',
                      'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
                      'rightShoe']

COLORS = np.array([
    (0, 0, 0), (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85), (0, 119, 221),
    (85, 85, 0), (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0), (0, 0, 255), (51, 170, 221),
    (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)
])

def dilate_mask(mask, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = (mask > 0).astype(np.uint8)
    return mask


# root_dir = "/Users/shukistern/PycharmProjects/stable-diffusion"
root_dir = "../.."
image_dir_name = 'target_ims'  # full body
images_dir = root_dir + f'/assets/sample_images/fashion_images/{image_dir_name}/frames'
parsing_dir = root_dir + f'/assets/sample_images/fashion_images/{image_dir_name}/human_parsing_crops'
# parsing_dir = root_dir + f'/assets/sample_images/fashion_images/{image_dir_name}/vis_human_parsing'

# cropped_save_dir = root_dir + f'/assets/sample_images/fashion_images/{image_dir_name}/cropped'
save_dir = root_dir + f'/assets/sample_images/fashion_images/{image_dir_name}/uncropped'
# os.makedirs(cropped_save_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
images = os.listdir(images_dir)
# mask_names = ["face", "hair"]
extensions = [".jpg", ".png", ".jpeg"]
images = [x for x in images if os.path.splitext(x)[1] in extensions]
mask_names = graphonomy_classes
H, W = 512, 512

for image_file in images:
    image_name = os.path.splitext(image_file)[0]
    image_suffix = os.path.splitext(image_file)[1]
    # if image_suffix not in [".jpg"]:
    #     continue
    image_path = os.path.join(images_dir, image_file)
    # parsing_paths = {mask_name: os.path.join(parsing_dir, f"{image_name}_{mask_name}.png") for mask_name in mask_names}
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    parse_dict_path = os.path.join(parsing_dir, f"{image_name}.npz")
    parse_dict = {**np.load(parse_dict_path)}

    masks = {}
    for i, cls in enumerate(graphonomy_classes):
        mask_i = parse_dict['parsing_mask']
        cls_mask = mask_i == i
        masks[cls] = cls_mask.astype(np.uint8)

    # masks = {mask_name: cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE) for mask_name, parsing_path in
    #          parsing_paths.items()}
    # masks = {mask_name: dilate_mask(mask, kernel_size=8) for mask_name, mask in masks.items()}
    masks["head"] = masks["face"] + masks["hair"]
    masks["head"] = (masks["head"] > 0).astype(np.uint8)
    masks["clothes"] = masks["upperclothes"] + masks["dress"] + masks["coat"] + masks["pants"] + \
                       masks["scarf"] + masks["skirt"] + masks["hat"] + masks["glove"] + masks["socks"] + \
                       masks["sunglasses"] + masks["leftShoe"] + masks["rightShoe"]
    masks["clothes"] = (masks["clothes"] > 0).astype(np.uint8)
    masks["skin"] = masks["face"] + masks["leftArm"] + masks["rightArm"] + masks["leftLeg"] + masks["rightLeg"] + masks["neck"]
    masks["skin"] = (masks["skin"] > 0).astype(np.uint8)
    masks["body"] = masks["skin"] + masks["hair"]
    masks["body"] = (masks["body"] > 0).astype(np.uint8)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(save_dir, f"{image_name}.png"), image)
    for mask_name in ["head", "clothes", "skin", "body", "face", "hair", "background"]:
        mask = masks[mask_name]
        cv2.imwrite(os.path.join(save_dir, f"{image_name}_{mask_name}.png"), (mask * 255).astype(np.uint8))

    #
    # img_width, img_height = image.shape[1], image.shape[0]
    # image = image[:H, (img_width - W) // 2:(img_width - W) // 2 + W]
    # masks = {mask_name: mask[:H, (img_width - W) // 2:(img_width - W) // 2 + W] for mask_name, mask in masks.items()}
    #
    # cv2.imwrite(os.path.join(cropped_save_dir, f"{image_name}.png"), image)
    # for mask_name in ["head", "clothes", "skin", "body", "face", "hair", "background"]:
    #     mask = masks[mask_name]
    #     cv2.imwrite(os.path.join(cropped_save_dir, f"{image_name}_{mask_name}.png"), (mask * 255).astype(np.uint8))
