
import cv2
import torch
import random
import copy
import numpy as np
from torchvision import transforms
from skimage.transform import resize
from torchvision.transforms import v2
from skimage.measure import regionprops, label
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from utils.affinity import multi_offset, gen_affs_ours, weight_binary_ratio  # Import from affinity.py


class NormalizeDivid255(object):
    """Normalize the image to the range [0, 1]."""
    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            sample['img'] = sample['img'] / 255.0
        return sample


class NormalizeMaxMin(object):
    """Normalize the image to the range [0, 1] using Min-Max normalization."""
    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:  ## and isinstance(sample['img'], np.ndarray)
            img = sample['img']
            min_val = img.min()
            max_val = img.max()
            if max_val > min_val:  # Avoid division by zero
                sample['img'] = (img - min_val) / (max_val - min_val)
            else:
                sample['img'] = np.zeros_like(img)  # or np.ones_like(img), depending on how you want to handle this case
        return sample



class InstanceToSemantic(object):
    """Convert instance segmentation masks to binary semantic masks."""
    def __call__(self, sample):
        if 'mask' in sample and sample['mask'] is not False:
            sample['mask'] = (sample['mask'] > 0).astype(np.uint8)
        return sample



class ToTensor(object):
    """Convert image and mask to Tensors if they are not already."""
    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            if isinstance(sample['img'], np.ndarray):
                sample['img'] = torch.from_numpy(sample['img'])
        if 'mask' in sample and sample['mask'] is not False:
            if isinstance(sample['mask'], np.ndarray):
                sample['mask'] = torch.from_numpy(sample['mask'])
        return sample


class FlipHorizontal(object):
    """Flip the image and mask horizontally."""
    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            sample['img'] = np.fliplr(sample['img']).copy()
        if 'mask' in sample and sample['mask'] is not False:
            sample['mask'] = np.fliplr(sample['mask']).copy()
        return sample

class FlipVertical(object):
    """Flip the image and mask vertically."""
    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            sample['img'] = np.flipud(sample['img']).copy()
        if 'mask' in sample and sample['mask'] is not False:
            sample['mask'] = np.flipud(sample['mask']).copy()
        return sample


class GrayToColor(object):
    """Convert a grayscale image to a 3-channel color image by repeating the grayscale values across the color channels.
       mask will remain unchanged.
    """

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            img = sample['img']
            if img.ndim == 2:  # If grayscale image with shape (h, w)
                img = img[np.newaxis, :, :]  # Add channel dimension to get shape (1, h, w)
                img = np.repeat(img, 3, axis=0)  # Repeat along the channel dimension to get shape (3, h, w)
            elif img.shape[0] == 1:  # If grayscale image with shape (1, h, w)
                img = np.repeat(img, 3, axis=0)  # Repeat along the channel dimension to get shape (3, h, w)
            sample['img'] = img

        if 'mask' in sample and sample['mask'] is not False:
            # If you need to do something with the mask, handle it here
            pass  # For now, the mask is not changed

        return sample
    


class RandomCrop(object):
    """Randomly crop the image and mask to a specified size."""
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False and 'mask' in sample and sample['mask'] is not False:
            img, mask = sample['img'], sample['mask']
            _, h, w = img.shape  # Get original height and width
            new_h, new_w = self.crop_size, self.crop_size

            if h > new_h and w > new_w:
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)

                sample['img'] = img[:, top: top + new_h, left: left + new_w]
                sample['mask'] = mask[:, top: top + new_h, left: left + new_w]

        return sample
    

class CenterCrop(object):
    """Crop the image and mask to a specified size from the center."""
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False and 'mask' in sample and sample['mask'] is not False:
            img, mask = sample['img'], sample['mask']
            _, h, w = img.shape  # Get original height and width
            new_h, new_w = self.crop_size, self.crop_size

            if h > new_h and w > new_w:
                top = (h - new_h) // 2
                left = (w - new_w) // 2

                sample['img'] = img[:, top: top + new_h, left: left + new_w]
                sample['mask'] = mask[:, top: top + new_h, left: left + new_w]

        return sample



class RandomInstanceSelection(object):
    """Randomly activate (highlight) at least a specified percentage of instances in the mask."""

    def __init__(self, min_percentage=0.6):
        """
        Initialize the RandomInstanceSelection with a minimum percentage.

        Args:
            min_percentage (float): The minimum percentage of instances to activate (between 0 and 1). Default is 0.6 (60%).
        """
        self.min_percentage = min_percentage

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False and 'mask' in sample and sample['mask'] is not False:
            img = sample['img']
            mask = sample['mask']

            # Ensure the mask has a single channel [1, w, h]
            if mask.shape[0] != 1:
                raise ValueError("Mask should have a single channel in shape [1, w, h].")

            # Remove the channel dimension for processing
            mask = mask[0]

            # Label each unique instance in the mask
            labeled_mask, num_instances = label(mask, return_num=True)
            instance_ids = np.unique(labeled_mask)

            # Skip background (instance_id 0)
            instance_ids = instance_ids[instance_ids != 0]

            # If no instances are found, return the original sample
            if len(instance_ids) == 0:
                return sample

            # Select at least min_percentage of the instances
            min_instances = max(1, int(self.min_percentage * len(instance_ids)))
            selected_instances = np.random.choice(instance_ids, size=min_instances, replace=False)

            # Create an empty mask for activated instances
            activated_mask = np.zeros_like(mask, dtype=np.uint8)

            for instance_id in selected_instances:
                # Create a mask for the selected instance
                instance_mask = (labeled_mask == instance_id).astype(np.uint8)

                # Activate the selected instance
                activated_mask[instance_mask > 0] = instance_id

            # Reshape the activated mask to have a channel dimension [1, w, h]
            activated_mask = activated_mask[np.newaxis, ...]

            # Update the sample with the activated mask
            sample['mask'] = activated_mask

        return sample




class RandomCompose(object):
    """Compose several transforms together with different probabilities."""
    def __init__(self, transforms, probabilities):
        assert len(transforms) == len(probabilities), "Transforms and probabilities must have the same length"
        self.transforms = transforms
        self.probabilities = probabilities

    def __call__(self, sample):
        for t, p in zip(self.transforms, self.probabilities):
            if random.random() < p:
                sample = t(sample)
        return sample


class Rotate:
    """
    Continuous rotation of the `xy`-plane.

    The sample size for `x`- and `y`-axes should be at least :math:`\sqrt{2}` times larger
    than the input size to make sure there is no non-valid region after center-crop.
    
    Args:
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, p=0.5):
        self.p = p
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT

    def rotate(self, img, M, interpolation):
        # Rotate each channel individually if multi-channel (3, w, h)
        if img.shape[0] == 1 or img.shape[0] == 3:  # Single or RGB image
            channels = [cv2.warpAffine(img[c], M, (img.shape[2], img.shape[1]), flags=interpolation, borderMode=self.border_mode) for c in range(img.shape[0])]
            transformed_img = np.stack(channels, axis=0)
        else:
            raise ValueError("Input image must have 1 or 3 channels.")
        return transformed_img

    def __call__(self, sample, random_state=np.random):
        if 'mask' in sample and sample['mask'] is not None:
            image, mask = sample['img'], sample['mask']
        else:
            image, mask = sample['img'], None

        height, width = image.shape[1:]  # get height and width from shape [1/3, w, h]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), random_state.rand() * 360.0, 1)

        sample['img'] = self.rotate(image, M, self.image_interpolation)
        if mask is not None:
            sample['mask'] = self.rotate(mask, M, self.label_interpolation)

        return sample
    

class ElasticTransform(object):
    """Apply elastic deformation to the image and mask with a given probability."""
    def __init__(self, alpha, sigma, p=0.2):
        self.elastic_transform = v2.ElasticTransform(alpha=alpha, sigma=sigma)
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            if 'img' in sample and sample['img'] is not False and 'mask' in sample and sample['mask'] is not False:
                img, mask = sample['img'], sample['mask']

                # Convert to float tensors
                img_tensor = torch.tensor(img, dtype=torch.float32)
                mask_tensor = torch.tensor(mask, dtype=torch.float32)

                # Combine img and mask to ensure the same transformation is applied
                combined = torch.cat([img_tensor, mask_tensor], dim=0)

                # Apply the elastic transform
                transformed = self.elastic_transform(combined)

                # Split the transformed image and mask
                img_transformed = transformed[0].unsqueeze(0)
                mask_transformed = transformed[1].unsqueeze(0)

                # Convert back to numpy arrays
                img_transformed = img_transformed.numpy()
                mask_transformed = mask_transformed.numpy()

                sample['img'], sample['mask'] = img_transformed, mask_transformed
        return sample
  

# class CopyPaste(object):
#     """Copy instances found in the mask and paste them to different background locations in the image and mask."""
#     def __init__(self, num_copies=1):
#         self.num_copies = num_copies

#     def __call__(self, sample):
#         if 'img' in sample and sample['img'] is not False and 'mask' in sample and sample['mask'] is not False:
#             img, mask = sample['img'], sample['mask']
#             h, w = img.shape[:2]

#             # Label the instances in the mask
#             labeled_mask = label(mask)
#             props = regionprops(labeled_mask)

#             if not props:
#                 return sample

#             # Number of instances to copy
#             num_to_copy = min(self.num_copies, len(props))

#             # Randomly select instances to copy
#             selected_props = random.sample(props, num_to_copy)

#             for prop in selected_props:
#                 bbox = prop.bbox
#                 # Ensure the bounding box has 4 values for 2D data
#                 if len(bbox) != 4:
#                     continue  # Skip this instance if it's not 2D
#                 min_row, min_col, max_row, max_col = bbox
#                 instance_img = img[min_row:max_row, min_col:max_col]
#                 instance_mask = mask[min_row:max_row, min_col:max_col]

#                 # Calculate distance transform to find less occupied areas
#                 dist_transform = distance_transform_edt(mask == 0)
#                 possible_locations = np.argwhere(dist_transform == np.max(dist_transform))

#                 # Select a random location to paste the instance
#                 paste_top_left_y, paste_top_left_x = possible_locations[random.randint(0, len(possible_locations) - 1)]
#                 paste_top_left_y = min(paste_top_left_y, h - (max_row - min_row))
#                 paste_top_left_x = min(paste_top_left_x, w - (max_col - min_col))

#                 # Ensure the selected area is background only
#                 while np.any(mask[paste_top_left_y:paste_top_left_y + (max_row - min_row), paste_top_left_x:paste_top_left_x + (max_col - min_col)]):
#                     possible_locations = np.argwhere(dist_transform == np.max(dist_transform))
#                     paste_top_left_y, paste_top_left_x = possible_locations[random.randint(0, len(possible_locations) - 1)]
#                     paste_top_left_y = min(paste_top_left_y, h - (max_row - min_row))
#                     paste_top_left_x = min(paste_top_left_x, w - (max_col - min_col))

#                 # Paste the instance at the new location
#                 img[paste_top_left_y:paste_top_left_y + (max_row - min_row), paste_top_left_x:paste_top_left_x + (max_col - min_col)] = instance_img
#                 mask[paste_top_left_y:paste_top_left_y + (max_row - min_row), paste_top_left_x:paste_top_left_x + (max_col - min_col)] = instance_mask

#             sample['img'], sample['mask'] = img, mask
#         return sample

class PadWithPaddingSize(object):
    """Pad the images and masks in a batch with a specific padding size on each side."""
    def __init__(self, padding_size, padding_mode='reflect', constant_values=0):
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.constant_values = constant_values

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            sample['img'] = self.pad(sample['img'])
        if 'mask' in sample and sample['mask'] is not False:
            sample['mask'] = self.pad(sample['mask'])
        return sample

    def pad(self, array_or_tensor):
        if isinstance(array_or_tensor, np.ndarray):
            if array_or_tensor.ndim not in [3, 4]:
                raise ValueError("Input numpy array must have 3 or 4 dimensions (1/3, w, h or batch_size, 1/3, w, h)")

            if array_or_tensor.ndim == 3:
                array_or_tensor = np.expand_dims(array_or_tensor, axis=0)

            padding = ((0, 0), (0, 0), (self.padding_size, self.padding_size), (self.padding_size, self.padding_size))
            
            if self.padding_mode == 'constant':
                padded_array = np.pad(array_or_tensor, padding, mode=self.padding_mode, constant_values=self.constant_values)
            else:
                padded_array = np.pad(array_or_tensor, padding, mode=self.padding_mode)
            
            if padded_array.shape[0] == 1:
                padded_array = np.squeeze(padded_array, axis=0)

            return padded_array

        elif torch.is_tensor(array_or_tensor):
            if array_or_tensor.dim() not in [3, 4]:
                raise ValueError("Input tensor must have 3 or 4 dimensions (1/3, w, h or batch_size, 1/3, w, h)")

            if array_or_tensor.dim() == 3:
                array_or_tensor = array_or_tensor.unsqueeze(0)

            padding = (self.padding_size, self.padding_size, self.padding_size, self.padding_size)

            if self.padding_mode == 'constant':
                padded_tensor = F.pad(array_or_tensor, padding, mode=self.padding_mode, value=self.constant_values)
            else:
                padded_tensor = F.pad(array_or_tensor, padding, mode=self.padding_mode)

            if padded_tensor.size(0) == 1:
                padded_tensor = padded_tensor.squeeze(0)

            return padded_tensor

        else:
            raise TypeError("Input should be either a numpy array or a torch tensor")



class Rescale(object):
    """
    Rescale augmentation.

    Args:
        low (float): lower bound of the random scale factor. Default: 0.8
        high (float): higher bound of the random scale factor. Default: 1.2
    """
    def __init__(self, low=0.8, high=1.2):
        self.low = low
        self.high = high
        self.image_interpolation = 1  # Bi-linear for images
        self.label_interpolation = 0  # Nearest for masks

    def random_scale(self, random_state):
        return random_state.rand() * (self.high - self.low) + self.low

    def apply_rescale_and_crop(self, img, sf, interpolation, original_size, random_state):
        # Rescale image
        rescaled_img = resize(img, 
                              (img.shape[0], int(img.shape[1] * sf), int(img.shape[2] * sf)), 
                              order=interpolation, mode='constant', cval=0, 
                              clip=True, preserve_range=True, anti_aliasing=(interpolation != 0))

        # Center crop to the original size
        center_y, center_x = rescaled_img.shape[1] // 2, rescaled_img.shape[2] // 2
        crop_y, crop_x = original_size

        start_y = max(0, center_y - crop_y // 2)
        start_x = max(0, center_x - crop_x // 2)

        end_y = start_y + crop_y
        end_x = start_x + crop_x

        cropped_img = rescaled_img[:, start_y:end_y, start_x:end_x]

        # Ensure the cropped image has the correct dimensions
        if cropped_img.shape[1] < crop_y or cropped_img.shape[2] < crop_x:
            padded_img = np.zeros((img.shape[0], crop_y, crop_x), dtype=img.dtype)
            padded_img[:, :cropped_img.shape[1], :cropped_img.shape[2]] = cropped_img
            cropped_img = padded_img

        return cropped_img

    def __call__(self, sample, random_state=np.random):
        if 'img' in sample and sample['img'] is not False and 'mask' in sample and sample['mask'] is not False:
            img = sample['img']
            mask = sample['mask']

            # Get original size
            original_size = img.shape[1], img.shape[2]

            # Get the same random scale factor for both image and mask
            sf = self.random_scale(random_state)

            # Apply rescale and crop to both image and mask
            sample['img'] = self.apply_rescale_and_crop(img, sf, self.image_interpolation, original_size, random_state)
            sample['mask'] = self.apply_rescale_and_crop(mask, sf, self.label_interpolation, original_size, random_state)

        return sample




class Grayscale:
    """Grayscale intensity augmentation, adapted from ELEKTRONN (http://elektronn.org/).

    Randomly adjust contrast/brightness, randomly invert the color space
    and apply gamma correction.

    Args:
        contrast_factor (float): intensity of contrast change. Default: 0.3
        brightness_factor (float): intensity of brightness change. Default: 0.3
        mode (string): one of ``'2D'``, ``'3D'`` or ``'mix'``. Default: ``'mix'``
        p (float): probability of applying the augmentation. Default: 0.5
    """

    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, mode='mix', p=0.5):
        """Initialize parameters."""
        assert 0.0 <= p <= 1.0, "Probability p must be between 0 and 1"
        self.CONTRAST_FACTOR = contrast_factor
        self.BRIGHTNESS_FACTOR = brightness_factor
        self.mode = mode
        self.p = p

    def __call__(self, sample, random_state=None):
        """Apply the grayscale augmentation."""
        if random_state is None:
            random_state = np.random
        
        if random_state.rand() < self.p:
            mode = self.mode
            if mode == 'mix':
                mode = '3D' if random_state.rand() > 0.5 else '2D'
            
            if mode == '2D':
                sample = self._augment2D(sample, random_state)
            elif mode == '3D':
                sample = self._augment3D(sample, random_state)

        return sample

    def _augment2D(self, sample, random_state):
        """Apply 2D grayscale augmentation."""
        imgs = sample['img'].astype(np.float32)  # Ensure floating-point operations
        transformed_imgs = np.copy(imgs)
        ran = random_state.rand(transformed_imgs.shape[-3] * 3)

        for z in range(transformed_imgs.shape[-3]):
            img = transformed_imgs[z, :, :]
            img *= 1 + (ran[z*3] - 0.5) * self.CONTRAST_FACTOR
            img += (ran[z*3+1] - 0.5) * self.BRIGHTNESS_FACTOR
            img = np.clip(img, 0, 1)
            img **= 2.0 ** (ran[z*3+2] * 2 - 1)
            transformed_imgs[z, :, :] = img

        sample['img'] = transformed_imgs.astype(sample['img'].dtype)  # Convert back to original dtype
        return sample

    def _augment3D(self, sample, random_state):
        """Apply 3D grayscale augmentation."""
        imgs = sample['img'].astype(np.float32)  # Ensure floating-point operations
        transformed_imgs = np.copy(imgs)
        ran = random_state.rand(3)

        transformed_imgs *= 1 + (ran[0] - 0.5) * self.CONTRAST_FACTOR
        transformed_imgs += (ran[1] - 0.5) * self.BRIGHTNESS_FACTOR
        transformed_imgs = np.clip(transformed_imgs, 0, 1)
        transformed_imgs **= 2.0 ** (ran[2] * 2 - 1)

        sample['img'] = transformed_imgs.astype(sample['img'].dtype)  # Convert back to original dtype
        return sample

    def _invert(self, sample):
        """Invert input images."""
        imgs = sample['img'].astype(np.float32)  # Ensure floating-point operations
        transformed_imgs = 1.0 - imgs
        transformed_imgs = np.clip(transformed_imgs, 0, 1)

        sample['img'] = transformed_imgs.astype(sample['img'].dtype)  # Convert back to original dtype
        return sample

    def _set_mode(self, mode):
        """Set 2D/3D/mix greyscale value augmentation mode."""
        assert mode in ['2D', '3D', 'mix'], "Mode must be '2D', '3D', or 'mix'"
        self.mode = mode


class Elastic(object):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    The implementation is based on https://gist.github.com/erniejunior/601cdf56d2b424757de5.

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

    Args:
        alpha (float): maximum pixel-moving distance of elastic deformation. Default: 10.0
        sigma (float): standard deviation of the Gaussian filter. Default: 4.0
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, alpha=16.0, sigma=4.0, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT

    def __call__(self, sample, random_state=None):
        if random_state is None:
            random_state = np.random

        if ('img' in sample and sample['img'] is not False and
            'mask' in sample and sample['mask'] is not False):
            
            image = sample['img']
            mask = sample['mask']

            height, width = image.shape[-2:]  # (c, y, x)

            dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha

            x, y = np.meshgrid(np.arange(width), np.arange(height))
            mapx, mapy = np.float32(x + dx), np.float32(y + dy)

            sample['img'] = self._apply_elastic_transform(image, mapx, mapy, self.image_interpolation)
            #sample['mask'] = self._apply_elastic_transform(mask, mapx, mapy, self.label_interpolation)

        return sample

    def _apply_elastic_transform(self, array, mapx, mapy, interpolation):
        transformed_array = []
        for i in range(array.shape[0]):
            transformed = cv2.remap(array[i], mapx, mapy, interpolation, borderMode=self.border_mode)
            transformed_array.append(transformed)

        transformed_array = np.stack(transformed_array, axis=0)
        return transformed_array



class CopyPaste(object):
    def __init__(self, num_copies=0.2):
        if isinstance(num_copies, float) and 0 < num_copies <= 1:
            self.num_copies_percentage = num_copies
            self.num_copies_fixed = None
        elif isinstance(num_copies, int) and num_copies > 0:
            self.num_copies_percentage = None
            self.num_copies_fixed = num_copies
        else:
            raise ValueError("num_copies should be a positive integer or a float between 0 and 1.")

    def random_rotation(self, img, mask):
        angle = random.choice([90, 180, 270])
        return np.rot90(img, k=angle//90, axes=(1, 2)), np.rot90(mask, k=angle//90, axes=(1, 2))

    def random_flip(self, img, mask):
        if random.choice([True, False]):
            img = np.flip(img, axis=2)
            mask = np.flip(mask, axis=2)
        if random.choice([True, False]):
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        return img, mask

    def random_rescale(self, img, mask, scale_range=(0.8, 1.2)):
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        h, w = img.shape[1:]  
        new_h, new_w = max(1, int(scale_factor * h)), max(1, int(scale_factor * w))

        if img.shape[0] == 1:
            img = cv2.resize(img[0], (new_w, new_h), interpolation=cv2.INTER_LINEAR)[np.newaxis, ...]
        else:
            img = cv2.resize(img.transpose(1, 2, 0), (new_w, new_h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        
        mask = cv2.resize(mask[0], (new_w, new_h), interpolation=cv2.INTER_NEAREST)[np.newaxis, ...]
        return img, mask

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False and 'mask' in sample and sample['mask'] is not False:
            img, mask = sample['img'], sample['mask']
            h, w = img.shape[1:]
            MaxAttempt = 15
            labeled_mask = label(mask[0])
            props = regionprops(labeled_mask)

            if not props:
                return sample

            num_to_copy = max(1, int(self.num_copies_percentage * len(props))) if self.num_copies_percentage is not None else min(self.num_copies_fixed, len(props))
            max_label = labeled_mask.max()

            selected_props = random.sample(props, num_to_copy)
            dist_transform = distance_transform_edt(mask[0] == 0)
            possible_locations = np.argwhere(dist_transform > 0)

            for prop in selected_props:
                bbox = prop.bbox
                if len(bbox) != 4:
                    continue
                min_row, min_col, max_row, max_col = bbox
                instance_img = img[:, min_row:max_row, min_col:max_col]
                instance_mask = (labeled_mask[min_row:max_row, min_col:max_col] == prop.label).astype(np.uint8)[np.newaxis, ...]

                instance_img, instance_mask = self.random_rotation(instance_img, instance_mask)
                instance_img, instance_mask = self.random_flip(instance_img, instance_mask)
                instance_img, instance_mask = self.random_rescale(instance_img, instance_mask)

                for _ in range(MaxAttempt):  # Limit attempts to find a suitable location
                    paste_top_left_y, paste_top_left_x = possible_locations[random.randint(0, len(possible_locations) - 1)]
                    if (paste_top_left_y + instance_img.shape[1] <= h and paste_top_left_x + instance_img.shape[2] <= w and
                        not np.any(mask[0, paste_top_left_y:paste_top_left_y + instance_mask.shape[1], paste_top_left_x:paste_top_left_x + instance_mask.shape[2]])):
                        max_label += 1
                        img[:, paste_top_left_y:paste_top_left_y + instance_img.shape[1], paste_top_left_x:paste_top_left_x + instance_img.shape[2]] = instance_img
                        mask[:, paste_top_left_y:paste_top_left_y + instance_mask.shape[1], paste_top_left_x:paste_top_left_x + instance_mask.shape[2]] = instance_mask * max_label
                        break

            sample['img'], sample['mask'] = img, mask
        return sample



class NormalizeDivid255(object):
    """Normalize the image to the range [0, 1]."""
    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            sample['img'] = sample['img'] / 255.0
        return sample


class RandomNoise(object):
    """Add random Gaussian noise to the image. Optionally add salt-and-pepper noise to the mask."""

    def __init__(self, mean, std_range=(0.01, 0.2), mask_noise=False, salt_pepper_prob=0.01):
        """
        Args:
            std_range (tuple): A tuple specifying the range of the standard deviation for the Gaussian noise. 
                               The std will be randomly chosen between these values for each call.
            mask_noise (bool): Whether to add noise to the mask. Currently not implemented.
            salt_pepper_prob (float): Probability for salt-and-pepper noise if mask_noise is True.
        """
        self.mean = mean  # Mean is always 0 for Gaussian noise
        self.std_range = std_range  # The range for standard deviation (randomized)
        self.mask_noise = mask_noise
        self.salt_pepper_prob = salt_pepper_prob

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            img = sample['img']
            original_shape = img.shape

            if img.shape[0] == 1:  # Grayscale image [1, w, h]
                img = img[0]  # Convert to [w, h]
            else:  # Color image [3, w, h]
                img = img.transpose(1, 2, 0)  # Convert to [w, h, 3]

            # Randomly sample a standard deviation within the provided range
            std = np.random.uniform(self.std_range[0], self.std_range[1])

            # Add Gaussian noise with mean 0 and random std
            noise = np.random.normal(self.mean, std, img.shape)
            noisy_img = np.clip(img + noise, 0, 1)  # Ensure the values stay within the range [0, 1]

            # Convert back to the original shape
            if original_shape[0] == 1:  # Grayscale image
                sample['img'] = noisy_img[np.newaxis, ...]  # Convert back to [1, w, h]
            else:  # Color image
                sample['img'] = noisy_img.transpose(2, 0, 1)  # Convert back to [3, w, h]

        if self.mask_noise and 'mask' in sample and sample['mask'] is not False:
            raise NotImplementedError("Noise to mask not implemented yet!")

        return sample
    

class RandomBlur(object):
    """Apply random Gaussian blur to the image."""

    def __init__(self, min_kernel_size=3, max_kernel_size=7):
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            img = sample['img']
            original_shape = img.shape

            if img.shape[0] == 1:  # Grayscale image [1, w, h]
                img = img[0]  # Convert to [w, h]
            else:  # Color image [3, w, h]
                img = img.transpose(1, 2, 0)  # Convert to [w, h, 3]

            # Apply Gaussian blur
            kernel_size = random.randint(self.min_kernel_size, self.max_kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure kernel size is odd
            blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

            if original_shape[0] == 1:  # Grayscale image
                sample['img'] = blurred_img[np.newaxis, ...]  # Convert back to [1, w, h]
            else:  # Color image
                sample['img'] = blurred_img.transpose(2, 0, 1)  # Convert back to [3, w, h]

        return sample
    

class Cutout(object):
    """Randomly mask out one or more square patches from the image. Optionally apply to the mask."""
    def __init__(self, max_num_holes=8, hole_size=8, fill_value=0, apply_to_mask=False):
        """
        Args:
            max_num_holes (int): Maximum number of patches to cut out of the image. 
                                 The actual number will be randomly chosen between 1 and max_num_holes.
            hole_size (int): The size (side length) of each square patch.
            fill_value (int or float): The value to fill the square patches (default is 0).
            apply_to_mask (bool): Whether to apply Cutout to the mask as well (default is False).
        """
        self.max_num_holes = max_num_holes
        self.hole_size = hole_size
        self.fill_value = fill_value
        self.apply_to_mask = apply_to_mask  # Control whether Cutout is applied to the mask

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            img = sample['img']
            h, w = img.shape[1:]  # Get height and width of the image (assume [C, H, W] format)

            # Randomly select the number of holes to apply (between 1 and max_num_holes)
            num_holes = np.random.randint(1, self.max_num_holes + 1)

            for _ in range(num_holes):
                # Randomly choose the top-left corner for the cutout patch
                y = np.random.randint(0, h - self.hole_size)
                x = np.random.randint(0, w - self.hole_size)

                # Fill the square patch with the specified fill_value in the image
                img[:, y:y + self.hole_size, x:x + self.hole_size] = self.fill_value

                # Optionally apply the cutout to the mask
                if self.apply_to_mask and 'mask' in sample and sample['mask'] is not False:
                    mask = sample['mask']
                    mask[:, y:y + self.hole_size, x:x + self.hole_size] = self.fill_value

            sample['img'] = img

        return sample


class GrayscaleJitter(object):
    """Apply random jittering to brightness and contrast of a grayscale image."""
    
    def __init__(self, brightness_factor=0.1, contrast_factor=0.1):
        """
        Args:
            brightness_factor (float): The range for brightness adjustment. The brightness will be
                                       adjusted by a factor randomly chosen between (1 - brightness_factor) 
                                       and (1 + brightness_factor).
            contrast_factor (float): The range for contrast adjustment. The contrast will be
                                     adjusted by a factor randomly chosen between (1 - contrast_factor)
                                     and (1 + contrast_factor).
        """
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def adjust_brightness(self, img, factor):
        """Adjust brightness by multiplying the image by the factor."""
        return np.clip(img * factor, 0, 1)  # Ensure values are within the valid range [0, 1]

    def adjust_contrast(self, img, factor):
        """Adjust contrast by shifting pixel values around the mean."""
        mean = img.mean()
        return np.clip((img - mean) * factor + mean, 0, 1)

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            img = sample['img']

            # Apply random brightness adjustment
            brightness_jitter = np.random.uniform(1 - self.brightness_factor, 1 + self.brightness_factor)
            img = self.adjust_brightness(img, brightness_jitter)

            # Apply random contrast adjustment
            contrast_jitter = np.random.uniform(1 - self.contrast_factor, 1 + self.contrast_factor)
            img = self.adjust_contrast(img, contrast_jitter)

            # Update the sample with the jittered image
            sample['img'] = img

        return sample


class AddLongStraightLineArtifact(object):
    """Add a single long straight line artifact (black line) across the perimeter of the image."""
    def __init__(self, thickness_range=(2, 8)):
        """
        Args:
            thickness_range (tuple): Range of thickness for the artifact line.
        """
        self.thickness_range = thickness_range

    def __call__(self, sample):
        if 'img' in sample and sample['img'] is not False:
            img = sample['img']
            _, h, w = img.shape  # Get the single-channel, height, and width of the image (now correctly unpacking)

            # Function to get a random point on the perimeter of the image
            def get_random_point_on_perimeter():
                side = random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top':
                    return random.randint(0, w - 1), 0  # (x, y)
                elif side == 'bottom':
                    return random.randint(0, w - 1), h - 1
                elif side == 'left':
                    return 0, random.randint(0, h - 1)
                else:  # 'right'
                    return w - 1, random.randint(0, h - 1)

            # Get two random points on opposite sides of the image
            x1, y1 = get_random_point_on_perimeter()
            x2, y2 = get_random_point_on_perimeter()
            while (x1 == 0 and x2 == 0) or (x1 == w - 1 and x2 == w - 1) or (y1 == 0 and y2 == 0) or (y1 == h - 1 and y2 == h - 1):
                x2, y2 = get_random_point_on_perimeter()  # Ensure points are on opposite sides

            # Generate random thickness for the artifact line
            thickness = random.randint(self.thickness_range[0], self.thickness_range[1])

            # Convert the image back to a 2D numpy array for OpenCV to work
            img_2d = img.squeeze()  # Remove the single-channel dimension

            # Draw the black line directly on the numpy array (float values between 0 and 1)
            img_with_artifacts = cv2.line(img_2d, (x1, y1), (x2, y2), color=0, thickness=thickness)

            # Add back the channel dimension after the line is added
            img_with_artifacts = img_with_artifacts[np.newaxis, :, :]

            sample['img'] = img_with_artifacts  # Replace the original image with the one containing the artifact

        return sample
    
######################################################################################
######################################################################################
######################################################################################
def apply_inference_augmentation():
    return transforms.Compose([
        NormalizeMaxMin(),
        InstanceToSemantic(),
        ToTensor()
    ])
######################################################################################
######################################################################################
######################################################################################
def apply_basic_train_augmentation(
                                crop_size = 512,
                                padding_size=None,
                                flip_horizontal_prob=0.2,
                                flip_vertical_prob=0.2,
                                rotate_prob=0.1,
                                rescale_prob=0.1,
                                ):
    """
    Applies a sequence of training augmentations to the data.

    Args:
        crop_size (tuple): The size to which the images will be randomly cropped.
        padding_size (int, optional): The size of padding to be applied. Defaults to None.
        flip_horizontal_prob (float): The probability of applying horizontal flip. Defaults to 0.5.
        flip_vertical_prob (float): The probability of applying vertical flip. Defaults to 0.5.
        add them all...
    Returns:
        transforms.Compose: A composition of the specified augmentations.
    """
    transform_list = [
        NormalizeMaxMin(),  # First mandatory transform
        RandomCompose(
            [
                FlipHorizontal(),
                FlipVertical(),
                Rotate(),
                Rescale(),
            ],
            [   
                flip_horizontal_prob,
                flip_vertical_prob,
                rotate_prob,
                rescale_prob,
            ]
        ),
         RandomCrop(crop_size),
         #CenterCrop(crop_size),
    ]

    # Insert PadWithPaddingSize if padding_size is specified
    if padding_size:
       transform_list.insert(1, PadWithPaddingSize(padding_size))

    return transforms.Compose(transform_list)

##############################################################################################
##############################################################################################
##############################################################################################
# Assuming RandomNoise, RandomBlur, Elastic, ToTensor, generate_affinity are defined elsewhere

def apply_train_augmentation(sample, offsets, separate_weight_affinity, mode='all', N=3):
    """Applies augmentations for both supervised and weakly supervised modes and returns a dictionary."""
    
    # Define augmentations for supervised learning
    augmentations_supervised = {
        "NoOp": lambda x: x,  # No operation, returns input unchanged
        "RandomNoise": RandomNoise(mean=0, std_range=(0.01, 0.1), mask_noise=False),
        "RandomBlur": RandomBlur(min_kernel_size=3, max_kernel_size=7),
        "Grayscale": Grayscale(),
        "AddLongStraightLineArtifact": AddLongStraightLineArtifact(),
        "Elastic": Elastic(),
        #"CopyPaste": CopyPaste(0.2),
        "Cutout": Cutout(),
        #"RandomInstanceSelection": RandomInstanceSelection(0.7)
    }

    # Define augmentations for weakly supervised learning
    augmentations_weakly_supervised = {
        "NoOp": lambda x: x,  # No operation, returns input unchanged, always applied
        "RandomNoise": RandomNoise(mean=0, std_range=(0.01, 0.1), mask_noise=False),
        "RandomBlur": RandomBlur(min_kernel_size=3, max_kernel_size=7),
        "Grayscale": Grayscale(),
        "AddLongStraightLineArtifact": AddLongStraightLineArtifact(),
        "Elastic": Elastic(),
        #"CopyPaste": CopyPaste(0.2),
        "Cutout": Cutout(),
        #"RandomInstanceSelection": RandomInstanceSelection(0.7),
    }

    # Apply augmentation process for both supervised and weakly supervised
    def apply_augmentations(augmentations, sample, mode, N):
        """Applies the given augmentations to the sample and returns augmented samples."""
        # Always apply "NoOp" as the first augmentation
        selected_augmentations = [("NoOp", augmentations.pop("NoOp"))]  # This removes "NoOp" from augmentations

        # Convert the remaining augmentations into a list of tuples
        augmentations_list = list(augmentations.items())

        # Apply random selection of augmentations if mode is 'random'
        if mode == 'random' and N is not None:
            N = min(max(N, 1), len(augmentations_list))  # Ensure valid N
            selected_augmentations.extend(random.sample(augmentations_list, N))
        else:
            # Apply all remaining augmentations if mode is 'all'
            selected_augmentations.extend(augmentations_list)  # "NoOp" is no longer in augmentations_list

        # Initialize lists to hold augmented images and masks
        img_list = []
        mask_list = []
        affs_mask_list = []
        labels_affs_list = []
        weight_map_list = []

        for aug_name, aug in selected_augmentations:
            sample_copy = copy.deepcopy(sample)  # Deep copy to avoid modifying the original

            # Apply augmentation
            augmented_sample = aug(sample_copy)

            # Ensure 'img' and 'mask' are center-cropped and converted to tensor
            augmented_sample = ToTensor()(augmented_sample)

            augmented_sample = generate_affinity(augmented_sample,
                                                 offsets,
                                                 separate_weight_affinity)

            # Append augmented 'img' and 'mask' to their respective lists
            img_list.append(augmented_sample['img'])
            mask_list.append(augmented_sample['mask'])
            affs_mask_list.append(augmented_sample['affs_mask'])
            labels_affs_list.append(augmented_sample['labels_affs'])

            # Convert numpy arrays in weight map to tensors before appending
            weight_map_tensor = torch.from_numpy(augmented_sample['weight_map'])
            weight_map_list.append(weight_map_tensor)

        # Stack all augmented images and masks into tensors of shape [augmentation_number, C, H, W]
        img_stack = torch.stack(img_list, dim=0)  # [augmentation_number, C, H, W]
        mask_stack = torch.stack(mask_list, dim=0)  # [augmentation_number, C, H, W]
        affs_mask_stack = torch.stack(affs_mask_list, dim=0)  # [augmentation_number, C, H, W]
        affs_labels_stack = torch.stack(labels_affs_list, dim=0)  # [augmentation_number, C, H, W]
        affs_weight_stack = torch.stack(weight_map_list, dim=0)  # [augmentation_number, C, H, W]

        # Prepare final output
        augmented_samples = {
            'img': img_stack,
            'mask': mask_stack,
            'affs_mask': affs_mask_stack,
            'labels_affs': affs_labels_stack,
            'weight_map': affs_weight_stack,
        }

        # Optionally create 'mask_semantic' for segmentation tasks
        if 'mask' in augmented_samples:
            augmented_samples['mask_semantic'] = (augmented_samples['mask'] > 0).long()

        return augmented_samples

    # Apply augmentations for both supervised and weakly supervised samples
    augmented_samples_supervised = apply_augmentations(augmentations_supervised, sample, mode='random', N=1)
    augmented_samples_weakly_supervised = apply_augmentations(augmentations_weakly_supervised, sample, mode='random', N=1)

    return {
        'aug_supervised': augmented_samples_supervised,
        'aug_weakly_supervised': augmented_samples_weakly_supervised,
    }




def generate_affinity(sample_index, offsets, separate_weight_affinity):
    """Calculate affinities and masks after transformations."""
    sample_index['labels_affs'], sample_index['affs_mask'] = gen_affs_ours(
        sample_index['mask'],
        offsets=offsets,
        ignore=False,
        padding=True
    )

    # Generate weight map
    if separate_weight_affinity:
        weight_map = np.zeros_like(sample_index['labels_affs'])
        for i in range(sample_index['labels_affs'].shape[0]):
            weight_map[i] = weight_binary_ratio(sample_index['labels_affs'][i], mask=sample_index['affs_mask'][i], alpha=1.0)
    else:
        weight_map = weight_binary_ratio(sample_index['labels_affs'], mask=sample_index['affs_mask'], alpha=1.0)

    sample_index['weight_map'] = weight_map

    return sample_index




