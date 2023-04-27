"""
Functions to work with images.
"""

import cv2
import numpy as np
import torch
from PIL import Image


def crop_img(image, i, j, h, w=None):
    """
    Crop and pad image with black pixels so that its' top-left corner is (i,j)

    :param image: Source image to crop
    :param i: Top coordinate
    :param j: Left coordinate
    :param h: Height
    :param w: Width
    :return: Cropped image
    """
    if w is None:
        w = h

    # Step 1: cut
    (i, j, h, w) = [int(x) for x in (i, j, h, w)]
    i_img = max(i, 0)
    j_img = max(j, 0)
    h_img = max(h + min(i, 0), 0)
    w_img = max(w + min(j, 0), 0)

    image = image[i_img:i_img + h_img, j_img:j_img + w_img]

    # Step 2: pad
    H_new, W_new, _ = image.shape

    pad_top = -min(i, 0)
    pad_left = -min(j, 0)
    pad_bot = max(h - H_new - pad_top, 0)
    pad_right = max(w - W_new - pad_left, 0)

    pad_seq = [(pad_top, pad_bot), (pad_left, pad_right)]
    if len(image.shape) == 3:
        pad_seq.append((0, 0))

    if (pad_top + pad_bot + pad_left + pad_right) > 0:
        image = np.pad(image, pad_seq, mode='constant')

    return image


def resize_image(image, shape):
    """
    Resize input image to target shape with PIL resize (less aliasing artifacts than OpenCV)

    :param image: Source image
    :param shape: Target shape of the image
    :return: Resized image
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if image.shape[-1] == 1:
        image = image[..., 0]  # workaround for single-channeled images

    image = np.asarray(Image.fromarray(image).resize(shape)) / 255.

    if len(image.shape) != 3:
        image = image[..., None]  # workaround for single-channeled images

    return image


def crop_resize_img(image, ltrb, target_size=512, nearest=False):
    """
    Crop by bounding box and resize to target size the image

    :param image: Source image
    :param ltrb: Bounding box set as (left, top, right, bottom)
    :param target_size: Target image size
    :param nearest: Legacy. Flag to use nearest interpolation
    :return: Cropped and resized image; Size of the cropped image; Crop mask for the image
    """
    l, t, r, b = ltrb
    sz = r - l
    tg_sizes = (target_size, target_size)
    crop_mask = np.ones_like(image)[..., :1].astype(np.float32)
    image = crop_img(image, t, l, sz)
    crop_mask = crop_img(crop_mask, t, l, sz)
    cropped_sizes = image.shape[:2]

    if nearest:
        image = cv2.resize(image, tg_sizes, cv2.INTER_NEAREST)
    else:
        image = resize_image(image, tg_sizes)
    crop_mask = resize_image(crop_mask, tg_sizes)[..., 0]
    crop_mask = (crop_mask == 1.).astype(np.float32)
    return image, cropped_sizes, crop_mask


def rgb2tensor(image, bbox, target_size):
    """
    Transform RGB image by bounding box and convert it into a tensor

    :param image: Input RGB image
    :param bbox: Bounding box set as (left, top, right, bottom)
    :param target_size: Target image size
    :return: Cropped and resized image; Cropped and resized crop mask; Size of the cropped area
    """
    image, cropped_sizes, crop_mask = crop_resize_img(image, bbox, target_size)

    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    cm_tensor = torch.tensor(crop_mask, dtype=torch.float32).unsqueeze(0)

    return image_tensor, cm_tensor, cropped_sizes


def segm2tensor(segmentation, bbox, target_size):
    """
    Transform segmentation map by bounding box and convert it into a tensor

    :param segmentation: Input segmentation map
    :param bbox: Bounding box set as (left, top, right, bottom)
    :param target_size: Target image size
    :return: Cropped and resized segmentation mask
    """
    segmentation, _, _ = crop_resize_img(segmentation, bbox, target_size)
    segmentation_tensor = torch.FloatTensor(segmentation).permute(2, 0, 1)

    segmentation_tensor = segmentation_tensor[:1]
    return segmentation_tensor


def get_square_bbox(ltrb):
    """
    Makes square bbox from any bbox by stretching of minimal length side

    :param ltrb: bbox tuple of size 4: input bbox (left, upper, right, lower)
    :return: tuple of size 4:  resulting square bbox (left, upper, right, lower)
    """
    left, upper, right, lower = ltrb
    width, height = right - left, lower - upper

    if width > height:
        y_center = (upper + lower) // 2
        upper = y_center - width // 2
        lower = upper + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, upper, right, lower


def scale_bbox(ltrb, scale):
    """
    Rescale bounding box size from the center with scale coefficient

    :param ltrb: Bounding box set with (left, top, right, bottom) coordinates
    :param scale: Scale coefficient
    :return: Rescaled bounding box
    """
    left, upper, right, lower = ltrb
    width, height = right - left, lower - upper

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def itt(img):
    """
    Image to Tensor. Transform an image into a tensor.
    
    :param img: Input image
    :return: Output tensor
    """
    tensor = torch.FloatTensor(img)  #
    if len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1)
    else:
        tensor = tensor.unsqueeze(0)
    return tensor


def tti(tensor):
    """
    Tensor to Image. Transform a tensor into an image.
    
    :param tensor: Input tensor
    :return: Output image
    """
    tensor = tensor.detach().cpu()
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    tensor = tensor.permute(1, 2, 0)
    image = tensor.numpy()
    if image.shape[-1] == 1:
        image = image[..., 0]
    return image


def get_lossmask(uv, real_segm, intersection=False):
    """
    Lossmask used to remove areas of SMPL-X model not covered by ground truth segmentation

    :param uv: UV-render of SMPL-X model
    :param real_segm: Ground truth segmentation map
    :return: Mask to apply to the output images to compute a loss
    """
    uv_mask = (uv > -1).sum(dim=1, keepdim=True) > 0
    real_segm = real_segm[:, :1]
    if intersection:
        lossmask = uv_mask.float() * real_segm.float()
    else:
        miss = (uv_mask.float() - real_segm.float()).clamp(0, 1)
        lossmask = 1. - miss
    return lossmask


def sample_values_by_uv(uv, rgb, texture_shape, segm=None):
    """
    Sample values from an RGB image back to the texture according to UV render
    
    Some tricks were taken here
    https://stackoverflow.com/questions/67990753/create-uv-texture-map-from-densepose-output
    
    :param uv: UV-render of SMPL-X model
    :param rgb: Donor RGB image 
    :param texture_shape: Output texture shape [batch_size, height, width]
    :param segm: Optional segmentation mask to know in which areas don't sample values
    :return: Three sampled textures: [RGB sampled texture; mask with known texels; mask with inpainted texels]
    """

    texture_size = texture_shape[3]
    batch_size = texture_shape[0]
    grid = np.zeros((batch_size, texture_size, texture_size, rgb.shape[1]), dtype=np.float32)
    counts = np.zeros((batch_size, texture_size, texture_size, rgb.shape[1]), dtype=np.uint8)
    masks = np.zeros((batch_size, 1, texture_size, texture_size), dtype=np.float32)
    masks_inpaint = np.zeros((batch_size, 1, texture_shape[2], texture_shape[3]), dtype=np.float32)

    uv = uv.detach().cpu().numpy()
    uv = np.transpose(uv, (0, 2, 3, 1))
    if segm is not None:
        segm = segm.detach().cpu().numpy()
        segm = np.transpose(segm, (0, 2, 3, 1))

    uv[..., 0] = (uv[..., 0] + 1) * texture_size / 2
    uv[..., 1] = (uv[..., 1] + 1) * texture_size / 2
    uv = np.around(uv).astype(np.int32)
    uv = np.clip(uv, 0, texture_size-1)

    for num in range(uv.shape[0]):
        if segm is not None:
            kernel = np.ones((5, 5), 'uint8')
            segm_erod = cv2.erode(segm[num], kernel, iterations=1)[..., None]
            y, x = np.where(np.any(uv[num] * segm_erod > 0, axis=-1))
        else:
            y, x = np.where(np.any(uv[num], axis=-1))

        pos = uv[num, y, x]

        pos[pos[:, 0] == 0] = 1
        pos[pos[:, 1] == 0] = 1

        pos[pos[:, 0] == texture_size-1] = texture_size-2
        pos[pos[:, 1] == texture_size-1] = texture_size-2

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                grid[num, pos[:, 1] + i, pos[:, 0] + j] += rgb[num, :, y, x]
                counts[num, pos[:, 1] + i, pos[:, 0] + j] += 1

        grid[num] /= counts[num]
        grid[num, pos[:, 1], pos[:, 0]] = rgb[num, :, y, x]

        mask = np.zeros((texture_size, texture_size), dtype=np.uint8)
        closed_mask = np.zeros((texture_size, texture_size), dtype=np.uint8)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                closed_mask[pos[:, 1] + i, pos[:, 0] + j] = 255
        mask[pos[:, 1], pos[:, 0]] = 255

        inpainting_mask = closed_mask - mask
        masks[num, 0, :, :] = closed_mask / 255
        masks_inpaint[num, 0, :, :] = inpainting_mask / 255

    grid = np.transpose(grid, (0, 3, 1, 2))
    grid = grid.astype(np.uint8)
    return grid, masks, masks_inpaint
