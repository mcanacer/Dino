import random
from PIL import Image, ImageFilter, ImageOps
from torchvision import datasets, transforms

import math
import torch
from torch.utils.data import default_collate

# https://github.com/facebookresearch/dino/blob/main/main_dino.py#L419

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ])

        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ])

        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class Collate(object):
    def __init__(
            self,
            image_size=224,
            num_global_crops=2,
            patch_size=16,
            pred_ratio=0.3,
            min_aspect_ratio=0.3,
            max_aspect_ratio=None,
            min_num_patches=4,
            max_num_patches=None,
            max_iterations=10,
    ):
        self.image_size = image_size
        self.num_global_crops = num_global_crops
        self.patch_size = patch_size
        self.pred_ratio = pred_ratio
        self.log_min_aspect_ratio = math.log(min_aspect_ratio)
        self.log_max_aspect_ratio = math.log(max_aspect_ratio or 1 / min_aspect_ratio)
        self.min_num_patches = min_num_patches
        self.max_num_patches = 0.5 * (image_size // patch_size) ** 2 if max_num_patches is None else max_num_patches
        self.max_iterations = max_iterations

    def _generate_mask(self, H, W):
        num_mask_patches = int(self.pred_ratio * H * W)
        mask = torch.zeros((H, W), dtype=torch.bool)
        mask_count = 0

        while mask_count < num_mask_patches:
            max_mask_patches = num_mask_patches - mask_count
            if max_mask_patches < self.min_num_patches:
                break

            current_max_patches = min(max_mask_patches, self.max_num_patches)
            for _ in range(self.max_iterations):
                area = random.randint(self.min_num_patches, int(current_max_patches))
                aspect_ratio = math.exp(random.uniform(self.log_min_aspect_ratio, self.log_max_aspect_ratio))
                h = int(round(math.sqrt(area * aspect_ratio)))
                w = int(round(math.sqrt(area / aspect_ratio)))

                if h < H and w < W:
                    top, left = random.randint(0, H - h), random.randint(0, W - w)
                    num_overlapping = mask[top: top + h, left: left + w].sum()
                    zeros_count = (h * w) - num_overlapping

                    if zeros_count > 0 and zeros_count <= max_mask_patches:
                        mask[top: top + h, left: left + w] = True
                        mask_count += zeros_count
                        break
            else:
                break
        return mask

    def __call__(self, batch):
        images_list, labels_list = zip(*batch)

        images = default_collate(images_list)

        collated_masks = []

        for i in range(self.num_global_crops):
            N, _, H_img, W_img = images[i].shape
            H, W = H_img // self.patch_size, W_img // self.patch_size

            batch_masks = []
            for _ in range(N):
                mask = self._generate_mask(H, W)
                batch_masks.append(mask)

            collated_masks.append(torch.stack(batch_masks))

        masks = torch.stack(collated_masks, dim=1)  # [N, 2, H, W]

        return images, masks
