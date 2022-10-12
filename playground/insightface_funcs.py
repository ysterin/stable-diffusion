import cv2
import math
# import matplotlib.pyplot as plt
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import torch
import kornia
from insightface.utils import face_align
from backbones import get_model

# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app = FaceAnalysis(providers=['CPUExecutionProvider'])
# app2 = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])

app.prepare(ctx_id=-1, det_size=(640, 640))  # 0
# app.prepare(ctx_id=0, det_size=(640, 640))  # 0


name = 'r100'
# model_weights_path = '/home/galgozes/insightface/model_zoo/ms1mv3_arcface_r50_fp16/backbone.pth'
model_weights_path = '../insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth'
net = get_model(name, fp16=True)
net.load_state_dict(torch.load(model_weights_path))
net.eval()


@torch.no_grad()
def get_init_feat(im):
    faces = app.get(im)

    best_face = None
    best_face_size = 500
    for face in faces:
        max_w = (np.max(face['kps'][:, 1]) - np.min(face['kps'][:, 1]))
        max_h = (np.max(face['kps'][:, 0]) - np.min(face['kps'][:, 0]))
        face_size = max_w * max_h
        if face_size > np.maximum(best_face_size, 0.4):
            best_face_size = face_size
            best_face = face

    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()
    print(f"len faces: {len(faces)}")
    if len(faces) == 0 or best_face is None:
        return None

    M, _ = face_align.estimate_norm(best_face['kps'], 112, 'arcface')
    warped = cv2.warpAffine(im, M, (112, 112), borderValue=0.0)

    plt_ims = False   # plt_ims = True
    if plt_ims:
        import matplotlib.pyplot as plt
        plt.imshow(warped)
        plt.show()
        return None

    # img_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    img_warped = np.transpose(warped, (2, 0, 1))
    img_warped = torch.from_numpy(img_warped).unsqueeze(0).float()
    img_warped.div_(255).sub_(0.5).div_(0.5)

    img_feats = net(img_warped).numpy()
    return M, img_feats


def get_feat(im, M, net):  # for inference
    im = ((im + 1) * 127.5).clamp(0, 255)
    # permute = [2, 1, 0]
    # im = im[:, permute]
    warped_im = kornia.geometry.transform.warp_affine(im, M.to(torch.float32), (112, 112), fill_value=0.0)
    warped_im.div_(255).sub_(0.5).div_(0.5)
    img_feats = net.to('cuda')(warped_im)
    return img_feats


def cos_loss_f(a, b):
    cos_loss = torch.einsum('...i,...i->...', a, b) / (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1))

    return cos_loss  # np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# von mises fisher dist


def margin_loss_f(labels, preds):  # from insightface  logits, labels
    s = 64
    m1 = 1.0
    m2 = 0.5
    m3 = 0.0

    index_positive = torch.where(labels != -1)[0]

    cos_m = math.cos(m2)
    sin_m = math.sin(m2)
    theta = math.cos(math.pi - m2)
    sinmm = math.sin(math.pi - m2) * m2

    sin_theta = torch.sqrt(1.0 - torch.pow(labels, 2))
    cos_theta_m = labels * cos_m - sin_theta * sin_m  # cos(target+margin)

    final_target_logit = torch.where(labels > theta, cos_theta_m, labels - sinmm)
    preds[index_positive, labels[index_positive].view(-1)] = final_target_logit
    logits = preds * s

    return logits


def cos_loss_f_numpy(a, b):
    cos_loss = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return cos_loss


def main():
    # '/home/galgozes/data/diffusion/CelebAMask-HQ/CelebA-HQ-img/1.jpg'
    path1 = '/home/galgozes/data/diffusion/botika/sapir.jpg'
    img1 = cv2.imread(path1)
    path2 = '/home/galgozes/data/diffusion/botika/sapir4.jpg'
    # path2 = '/home/galgozes/data/diffusion/CelebAMask-HQ/CelebA-HQ-img/4.jpg'
    img2 = cv2.imread(path2)
    # img2 = img1[:, ::-1, :]
    _, img_feats = get_init_feat(img1[..., ::-1])
    _, img_feats2 = get_init_feat(img2[..., ::-1])

    cos_loss = cos_loss_f_numpy(img_feats[0], img_feats2[0])
    tau = 1
    constractive_loss = -np.log((np.exp(cos_loss / tau) / np.sum(np.exp(cos_loss / tau))))
    g = 1

    # import matplotlib.pyplot as plt
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()


if __name__ == "__main__":
    main()

