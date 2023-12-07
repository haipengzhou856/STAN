import random
import numpy as np
from PIL import Image
from torchvision import transforms


def get_train_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize(scale),
        RandomHorizontallyFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return joint_transform


def get_val_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize(scale),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return joint_transform




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, masks, darks):
        assert len(imgs) == len(masks)
        for t in self.transforms:
            imgs, masks, darks = t(imgs, masks, darks)
        return imgs, masks, darks


class RandomHorizontallyFlip(object):
    def __call__(self, imgs, masks, darks):
        if random.random() < 0.5:
            for idx in range(len(imgs)):
                imgs[idx] = imgs[idx].transpose(Image.FLIP_LEFT_RIGHT)
                masks[idx] = masks[idx].transpose(Image.FLIP_LEFT_RIGHT)
                darks[idx] = darks[idx].transpose(Image.FLIP_LEFT_RIGHT)

        return imgs, masks, darks


class Resize(object):
    def __init__(self, scale):
        assert scale[0] <= scale[1]
        self.scale = scale

    def __call__(self, imgs, masks, darks):
        w, h = imgs[0].size
        for idx in range(len(imgs)):
            if w > h:
                imgs[idx] = imgs[idx].resize((self.scale[1], self.scale[0]), Image.BILINEAR)
                masks[idx] = masks[idx].resize((self.scale[1], self.scale[0]), Image.NEAREST)
                darks[idx] = darks[idx].resize((self.scale[1], self.scale[0]), Image.NEAREST)
            else:
                imgs[idx] = imgs[idx].resize((self.scale[0], self.scale[1]), Image.BILINEAR)
                masks[idx] = masks[idx].resize((self.scale[0], self.scale[1]), Image.NEAREST)
                darks[idx] = darks[idx].resize((self.scale[0], self.scale[1]), Image.NEAREST)

        return imgs, masks, darks


class ToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, imgs, masks, darks):
        for idx in range(len(imgs)):
            img_np = np.array(imgs[idx])
            mask_np = np.array(masks[idx])
            dark_np = np.array(darks[idx])

            x, y, _ = img_np.shape
            # make sure x is shorter than y
            if x > y:
                img_np = np.swapaxes(img_np, 0, 1)
                mask_np = np.swapaxes(mask_np, 0, 1)
                dark_np = np.swapaxes(dark_np, 0, 1)

            imgs[idx] = self.totensor(img_np)
            masks[idx] = self.totensor(mask_np).long()
            darks[idx] = self.totensor(dark_np)

        return imgs, masks,darks


class Normalize(object):
    def __init__(self, mean, std):
        self.normlize = transforms.Normalize(mean, std)

    def __call__(self, imgs, masks, darks):
        for idx in range(len(imgs)):
            imgs[idx] = self.normlize(imgs[idx])
        return imgs, masks, darks
