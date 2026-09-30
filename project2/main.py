# CS180 Project 2

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import skimage as sk
import skimage.io as skio
from PIL import Image, ImageOps
from scipy.signal import convolve2d

from align_image_code import (align_image_centers, align_images,
                              match_img_size, rescale_images, rotate_im1)


data_dir = Path(__file__).resolve().parent / 'data'
output_dir = Path(__file__).resolve().parent / 'output'
output_dir.mkdir(exist_ok=True)

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])


def save(image, filename, folder):
    image = np.clip(image, 0, 1)
    folder = output_dir / folder
    folder.mkdir(exist_ok=True)
    skio.imsave(folder / filename, sk.img_as_ubyte(image))


def center_crop(image, shape):
    height, width = shape
    start_y = (image.shape[0] - height) // 2
    start_x = (image.shape[1] - width) // 2
    return image[start_y:start_y + height, start_x:start_x + width]


# Part 1.1: Convolutions from Scratch
def convolve_four_loops(image, filter, mode='same'):
    flipped_filter = np.flip(np.flip(filter, axis=0), axis=1)
    image_h, image_w = image.shape
    fh, fw = flipped_filter.shape

    if mode == 'same':
        pad_top = fh // 2
        pad_bottom = (fh - 1) // 2
        pad_left = fw // 2
        pad_right = (fw - 1) // 2

    elif mode == 'full':
        pad_top = pad_bottom = fh - 1
        pad_left = pad_right = fw - 1

    else:
        raise ValueError("mode must be 'same' or 'full'")

    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )

    output_h = padded.shape[0] - (fh - 1)
    output_w = padded.shape[1] - (fw - 1)


    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            for fi in range(fh):
                for fj in range(fw):
                    output[i, j] += (padded[i + fi, j + fj] * flipped_filter[fi, fj])

    return output


def convolve_two_loops(image, filter, mode = 'same'):
    flipped_filter = np.flip(np.flip(filter, axis=0), axis=1)
    image_h, image_w = image.shape
    fh, fw = flipped_filter.shape

    if mode == 'same':
        pad_top = fh // 2
        pad_bottom = (fh - 1) // 2
        pad_left = fw // 2
        pad_right = (fw - 1) // 2

    elif mode == 'full':
        pad_top = pad_bottom = fh - 1
        pad_left = pad_right = fw - 1

    else:
        raise ValueError("mode must be 'same' or 'full'")

    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )

    output_h = padded.shape[0] - (fh - 1)
    output_w = padded.shape[1] - (fw - 1)


    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            patch = padded[i:i+fh, j:j+fw]
            output[i, j] = np.sum(patch * flipped_filter)

    return output

# Part 1.2: Finite Difference Operator
def finite_difference(image, threshold=0.28):
    dx_result = convolve2d(image, Dx, mode='same', boundary='symm')
    dy_result = convolve2d(image, Dy, mode='same', boundary='symm')

    gradient_magnitude = np.sqrt(dx_result**2 + dy_result**2)

    dx_display = (dx_result - dx_result.min()) / (dx_result.max() - dx_result.min())
    dy_display = (dy_result - dy_result.min()) / (dy_result.max() - dy_result.min())
    gradient_display = gradient_magnitude / gradient_magnitude.max()
    edges = gradient_display > threshold

    save(image, 'cameraman.jpg', 'part1_2')
    save(dx_display, 'cameraman_dx.jpg', 'part1_2')
    save(dy_display, 'cameraman_dy.jpg', 'part1_2')
    save(gradient_display, 'cameraman_gradient.jpg', 'part1_2')
    save(edges.astype(float), 'cameraman_edges.png', 'part1_2')
    for test_threshold in (0.10, 0.28, 0.50):
        test_edges = gradient_display > test_threshold
        save(test_edges.astype(float), f'edges_{test_threshold}.png', 'part1_2')

    return dx_result, dy_result, gradient_magnitude, edges



# Part 1.3: Derivative of Gaussian Filter
def derivative_of_gaussian(image, kernel_size=9, sigma=2, threshold=0.28):
    # create the Gaussian filter
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian = gaussian_1d @ gaussian_1d.T

    # pad once so both methods use the same pixels at the image boundary
    pad = kernel_size // 2 + 1
    padded_image = np.pad(image, pad, mode='symmetric')

    # method 1: use full convolutions and crop once at the end
    blurred_full = convolve2d(padded_image, gaussian, mode='full')
    blurred_dx_full = convolve2d(blurred_full, Dx, mode='full')
    blurred_dy_full = convolve2d(blurred_full, Dy, mode='full')

    blurred = center_crop(blurred_full, image.shape)
    blurred_dx = center_crop(blurred_dx_full, image.shape)
    blurred_dy = center_crop(blurred_dy_full, image.shape)

    blurred_gradient = np.sqrt(
        blurred_dx**2 + blurred_dy**2
    )

    # method 2: construct DoG filters first
    dog_x = convolve2d(gaussian, Dx, mode='full')
    dog_y = convolve2d(gaussian, Dy, mode='full')

    dog_dx_result = center_crop(
        convolve2d(padded_image, dog_x, mode='full'), image.shape
    )
    dog_dy_result = center_crop(
        convolve2d(padded_image, dog_y, mode='full'), image.shape
    )

    dog_gradient = np.sqrt(
        dog_dx_result**2 + dog_dy_result**2
    )

    # use the same scale so the two results are displayed the same way
    max_gradient = max(blurred_gradient.max(), dog_gradient.max())
    blurred_gradient_display = blurred_gradient / max_gradient
    dog_gradient_display = dog_gradient / max_gradient

    blurred_edges = blurred_gradient_display > threshold
    dog_edges = dog_gradient_display > threshold

    # normalize the filters so negative values can be displayed
    dog_x_display = (
        (dog_x - dog_x.min()) /
        (dog_x.max() - dog_x.min())
    )

    dog_y_display = (
        (dog_y - dog_y.min()) /
        (dog_y.max() - dog_y.min())
    )

    # save results
    save(blurred, 'cameraman_blurred.jpg', 'part1_3')
    save(blurred_gradient_display, 'blurred_gradient.jpg', 'part1_3')
    save(blurred_edges.astype(float), 'blurred_edges.png', 'part1_3')

    save(gaussian / gaussian.max(), 'gaussian_filter.png', 'part1_3')
    save(dog_x_display, 'dog_x_filter.png', 'part1_3')
    save(dog_y_display, 'dog_y_filter.png', 'part1_3')
    save(dog_gradient_display, 'dog_gradient.jpg', 'part1_3')
    save(dog_edges.astype(float), 'dog_edges.png', 'part1_3')

    return blurred_gradient, blurred_edges, dog_gradient, dog_edges

# Part 2.1: Image Sharpening

def filter_image(image, filter):
    image = sk.img_as_float(image)

    if image.ndim == 2:
        return convolve2d(image, filter, mode='same', boundary='symm')

    result = np.zeros_like(image, dtype=float)
    for channel in range(image.shape[2]):
        result[:, :, channel] = convolve2d(image[:, :, channel], filter, mode='same', boundary='symm')

    return result

def sharpen_image(image, kernel_size=9, sigma=2, alpha=1.5):
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian = gaussian_1d @ gaussian_1d.T

    unsharp_mask_filter = np.zeros_like(gaussian)
    unsharp_mask_filter[kernel_size//2, kernel_size//2] = 1 + alpha
    unsharp_mask_filter -= alpha * gaussian

    sharpened_image = filter_image(image, unsharp_mask_filter)
    return np.clip(sharpened_image, 0, 1)


def save_sharpening_results(image, name, kernel_size=9, sigma=2,
                            alphas=(0.5, 1.5, 3), save_resharpened=True):
    image = sk.img_as_float(image)
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian = gaussian_1d @ gaussian_1d.T
    blurred = filter_image(image, gaussian)
    high_frequency = image - blurred

    save(image, f'{name}_original.jpg', 'part2_1')
    save(blurred, f'{name}_blurred.jpg', 'part2_1')
    save(high_frequency + 0.5, f'{name}_high_frequency.jpg', 'part2_1')

    for alpha in alphas:
        sharpened = sharpen_image(image, kernel_size, sigma, alpha)
        save(sharpened, f'{name}_sharpened_alpha_{alpha}.jpg', 'part2_1')

    if save_resharpened:
        resharpened = sharpen_image(blurred, kernel_size, sigma, 1.5)
        save(resharpened, f'{name}_blurred_then_resharpened.jpg', 'part2_1')


# Part 2.2: Hybrid Images

def low_pass(image, sigma):
    kernel_size = 2 * int(3 * sigma) + 1
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    image = sk.img_as_float(image)
    return cv2.sepFilter2D(
        image, -1, gaussian_1d, gaussian_1d,
        borderType=cv2.BORDER_REFLECT
    )

def high_pass(image, sigma):
    return image - low_pass(image, sigma)


def hybrid_image(high_image, low_image, high_sigma, low_sigma,
                 high_weight=1):
    high_frequency = high_pass(high_image, high_sigma)
    low_frequency = low_pass(low_image, low_sigma)

    hybrid = high_weight * high_frequency + low_frequency
    hybrid = np.clip(hybrid, 0, 1)

    return hybrid, high_frequency, low_frequency


def fourier_spectrum(image):
    if image.ndim == 3:
        image = sk.color.rgb2gray(image[:, :, :3])

    spectrum = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(image))))
    return spectrum / spectrum.max()


def crop_alignment_borders(image1, image2):
    valid1 = np.max(image1[:, :, :3], axis=2) > 0.001
    valid2 = np.max(image2[:, :, :3], axis=2) > 0.001
    rows, columns = np.where(valid1 & valid2)

    if len(rows) == 0:
        return image1, image2

    top, bottom = rows.min(), rows.max() + 1
    left, right = columns.min(), columns.max() + 1
    return (image1[top:bottom, left:right],
            image2[top:bottom, left:right])


def save_hybrid_results(image1, image2, name, high_sigma, low_sigma,
                        points=None, crop=None, already_aligned=False,
                        save_analysis=False, high_weight=1, originals=None):
    if already_aligned:
        image1_aligned, image2_aligned = image1, image2
    elif points is None:
        image1_aligned, image2_aligned = align_images(image1, image2)
    else:
        image1_aligned, image2_aligned = align_image_centers(
            image1, image2, points
        )
        image1_aligned, image2_aligned = rescale_images(
            image1_aligned, image2_aligned, points
        )
        image1_aligned, _ = rotate_im1(
            image1_aligned, image2_aligned, points
        )
        image1_aligned, image2_aligned = match_img_size(
            image1_aligned, image2_aligned
        )

    if crop is None:
        image1_aligned, image2_aligned = crop_alignment_borders(
            image1_aligned, image2_aligned
        )
    else:
        top, bottom, left, right = crop
        image1_aligned = image1_aligned[top:bottom, left:right]
        image2_aligned = image2_aligned[top:bottom, left:right]
    hybrid, high_frequency, low_frequency = hybrid_image(
        image1_aligned, image2_aligned, high_sigma, low_sigma,
        high_weight
    )

    source1, source2 = (image1, image2) if originals is None else originals
    save(source1, f'{name}_high_original.jpg', 'part2_2')
    save(source2, f'{name}_low_original.jpg', 'part2_2')
    save(hybrid, f'{name}_hybrid.jpg', 'part2_2')
    save(resize_image(hybrid, (96, 96)), f'{name}_small.jpg', 'part2_2')

    if save_analysis:
        save(image1_aligned, f'{name}_high_aligned.jpg', 'part2_2')
        save(image2_aligned, f'{name}_low_aligned.jpg', 'part2_2')
        save(high_frequency + 0.5, f'{name}_high_frequency.jpg', 'part2_2')
        save(low_frequency, f'{name}_low_frequency.jpg', 'part2_2')
        save(fourier_spectrum(image1_aligned), f'{name}_high_original_fft.jpg', 'part2_2')
        save(fourier_spectrum(image2_aligned), f'{name}_low_original_fft.jpg', 'part2_2')
        save(fourier_spectrum(high_frequency), f'{name}_high_frequency_fft.jpg', 'part2_2')
        save(fourier_spectrum(low_frequency), f'{name}_low_frequency_fft.jpg', 'part2_2')
        save(fourier_spectrum(hybrid), f'{name}_hybrid_fft.jpg', 'part2_2')

    return hybrid


# Part 2.3: Gaussian and Laplacian Stacks

def gaussian_stack(image, levels=6, sigma=2):
    stack = [sk.img_as_float(image)]
    for i in range(levels - 1):
        stack.append(low_pass(stack[-1], sigma * (2**i)))
    return stack


def laplacian_stack(image, levels=6, sigma=2):
    gaussian = gaussian_stack(image, levels, sigma)
    laplacian = []

    for i in range(levels - 1):
        laplacian.append(gaussian[i] - gaussian[i + 1])

    laplacian.append(gaussian[-1])
    return laplacian


def save_stacks(image, name, levels=6, sigma=2):
    gaussian = gaussian_stack(image, levels, sigma)
    laplacian = laplacian_stack(image, levels, sigma)

    for i in range(levels):
        save(gaussian[i], f'{name}_gaussian_{i}.jpg', 'part2_3')
        if i == levels - 1:
            save(laplacian[i], f'{name}_laplacian_{i}.jpg', 'part2_3')
        else:
            save(laplacian[i] + 0.5, f'{name}_laplacian_{i}.jpg', 'part2_3')


# Part 2.4: Multiresolution Blending

def multiresolution_blend(image1, image2, mask, levels=6, sigma=2):
    image1 = sk.img_as_float(image1)
    image2 = sk.img_as_float(image2)
    mask = np.asarray(mask, dtype=float)

    laplacian1 = laplacian_stack(image1, levels, sigma)
    laplacian2 = laplacian_stack(image2, levels, sigma)
    mask_stack = gaussian_stack(mask, levels, sigma)
    blended_stack = []

    for i in range(levels):
        level_mask = mask_stack[i]
        if image1.ndim == 3:
            level_mask = level_mask[:, :, np.newaxis]
        blended = level_mask * laplacian1[i]
        blended += (1 - level_mask) * laplacian2[i]
        blended_stack.append(blended)

    result = np.sum(blended_stack, axis=0)
    return np.clip(result, 0, 1), blended_stack, mask_stack


def vertical_mask(shape):
    mask = np.zeros(shape)
    mask[:, :shape[1] // 2] = 1
    return mask


def horizontal_mask(shape):
    mask = np.zeros(shape)
    mask[:shape[0] // 2, :] = 1
    return mask


def resize_image(image, shape):
    return sk.transform.resize(
        image, (shape[0], shape[1]), anti_aliasing=True
    )


def load_small_image(path, max_size=800):
    with Image.open(path) as pil_image:
        image = np.array(ImageOps.exif_transpose(pil_image).convert('RGB'))

    scale = min(1, max_size / max(image.shape[:2]))
    shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    return resize_image(image, shape)


def load_cropped_image(path, shape, box=None):
    with Image.open(path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
        size = (shape[1], shape[0])
        if box is None:
            pil_image = ImageOps.fit(pil_image, size, method=Image.Resampling.LANCZOS)
        else:
            pil_image = pil_image.crop(box).resize(size, Image.Resampling.LANCZOS)
        return sk.img_as_float(np.array(pil_image))


def save_blend_results(image1, image2, mask, name, levels=6, sigma=2,
                       save_process=False):
    image2 = resize_image(image2, image1.shape[:2])
    mask = resize_image(np.asarray(mask, dtype=float), image1.shape[:2])
    result, blended_stack, mask_stack = multiresolution_blend(
        image1, image2, mask, levels, sigma
    )

    save(image1, f'{name}_image1.jpg', 'part2_4')
    save(image2, f'{name}_image2.jpg', 'part2_4')
    save(mask, f'{name}_mask.jpg', 'part2_4')
    save(result, f'{name}.jpg', 'part2_4')
    if name != 'oraple':
        hard_mask = (mask > 0.5).astype(float)[:, :, np.newaxis]
        hard_cut = hard_mask * image1 + (1 - hard_mask) * image2
        save(hard_cut, f'{name}_hard_cut.jpg', 'part2_4')

    if save_process:
        # Szeliski Figure 3.42 layout. Rows: Laplacian levels 0, 2, 4, then
        # the full reconstruction (all levels summed). Columns: masked
        # image1, masked image2, and their sum.
        laplacian1 = laplacian_stack(image1, levels, sigma)
        laplacian2 = laplacian_stack(image2, levels, sigma)
        masked1, masked2 = [], []
        for i in range(levels):
            level_mask = mask_stack[i][:, :, np.newaxis]
            masked1.append(level_mask * laplacian1[i])
            masked2.append((1 - level_mask) * laplacian2[i])

        prefix = 'figure_3_42' if name == 'oraple' else f'{name}_process'
        folder = 'part2_3' if name == 'oraple' else 'part2_4'
        letters = iter('abcdefghijkl')
        for level in (0, 2, 4):
            row = [masked1[level], masked2[level], blended_stack[level]]
            for band in row:
                save(band + 0.5, f'{prefix}_{next(letters)}.jpg', folder)
        for image in (np.sum(masked1, axis=0), np.sum(masked2, axis=0), result):
            save(image, f'{prefix}_{next(letters)}.jpg', folder)

    return result

# Part 1.1 results
image = skio.imread(data_dir / 'personal' / 'kevin.jpeg', as_gray=True)
save(image, 'personal_original.jpg', 'part1_1')
box_filter = np.ones((9, 9)) / 81

small_image = image[:50, :50]
start = perf_counter()
four_loop_test = convolve_four_loops(small_image, box_filter)
four_loop_time = perf_counter() - start

start = perf_counter()
two_loop_test = convolve_two_loops(small_image, box_filter)
two_loop_time = perf_counter() - start

start = perf_counter()
scipy_test = convolve2d(small_image, box_filter, mode='same')
scipy_time = perf_counter() - start

print('Four loops:', round(four_loop_time, 4), 'seconds')
print('Two loops:', round(two_loop_time, 4), 'seconds')
print('SciPy:', round(scipy_time, 4), 'seconds')
print('Results match:', np.allclose(four_loop_test, scipy_test) and
      np.allclose(two_loop_test, scipy_test))

my_same = convolve_two_loops(image, box_filter, mode='same')
my_full = convolve_two_loops(image, box_filter, mode='full')
dx_result = convolve_two_loops(image, Dx, mode='same')
dy_result = convolve_two_loops(image, Dy, mode='same')

dx_display = (dx_result - dx_result.min()) / (dx_result.max() - dx_result.min())
dy_display = (dy_result - dy_result.min()) / (dy_result.max() - dy_result.min())

save(my_same, 'box_filter_same.jpg', 'part1_1')
save(my_full, 'box_filter_full.jpg', 'part1_1')
save(dx_display, 'dx.jpg', 'part1_1')
save(dy_display, 'dy.jpg', 'part1_1')

# Parts 1.2 and 1.3 results
cameraman = skio.imread(data_dir / 'cameraman.png', as_gray=True)
# The downloaded image has a white frame that would become a false edge.
cameraman = cameraman[7:-1, 1:-5]
finite_difference(cameraman)
derivative_of_gaussian(cameraman, kernel_size=9, sigma=2, threshold=0.28)

# Part 2.1 results
taj = skio.imread(data_dir / 'taj.jpg')
save_sharpening_results(taj, 'taj')
# The full-moon photo starts out soft, so sharpening visibly brings out craters.
moon_sharpen = load_small_image(data_dir / 'downloaded' / 'moon.jpg', max_size=960)
save_sharpening_results(
    moon_sharpen, 'moon', alphas=(1.5, 3), save_resharpened=False
)
top, bottom, left, right = 300, 640, 300, 640
save(moon_sharpen[top:bottom, left:right], 'moon_zoom_original.jpg', 'part2_1')
for alpha in (1.5, 3):
    save(sharpen_image(moon_sharpen, alpha=alpha)[top:bottom, left:right],
         f'moon_zoom_alpha_{alpha}.jpg', 'part2_1')

# Part 2.2 results
def align_by_eyes(image, eyes, target_eyes, shape):
    # Rotate, scale, and shift image so its two eye points land on target_eyes.
    transform, _ = cv2.estimateAffinePartial2D(
        np.float32(eyes), np.float32(target_eyes)
    )
    return cv2.warpAffine(image, transform, (shape[1], shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


derek = sk.img_as_float(skio.imread(data_dir / 'DerekPicture.jpg'))
nutmeg = sk.img_as_float(skio.imread(data_dir / 'nutmeg.jpg'))
nutmeg_aligned = align_by_eyes(
    nutmeg, [(610, 285), (753, 359)], [(298, 343), (438, 330)], derek.shape
)
save_hybrid_results(
    nutmeg_aligned[:720], derek[:720], 'derek_nutmeg', high_sigma=3,
    low_sigma=8, already_aligned=True, originals=(nutmeg, derek)
)

owl_path = data_dir / 'downloaded' / 'owl.jpg'
moon_path = data_dir / 'downloaded' / 'moon.jpg'
owl = sk.img_as_float(skio.imread(owl_path))
moon = sk.img_as_float(skio.imread(moon_path))
owl_aligned = load_cropped_image(owl_path, (480, 480), (40, 0, 540, 500))
moon_aligned = load_cropped_image(moon_path, (480, 480))
# Outside the lunar disc, replace the owl with its own blur so its
# high-pass is near zero there and no feathers spill into the black sky.
moon_disc = low_pass((moon_aligned.mean(axis=2) > 0.08).astype(float), 6)
moon_disc = moon_disc[:, :, np.newaxis]
owl_aligned = moon_disc * owl_aligned + (1 - moon_disc) * low_pass(owl_aligned, 2)
save_hybrid_results(
    owl_aligned, moon_aligned, 'owl_moon', high_sigma=2, low_sigma=6,
    high_weight=0.7, already_aligned=True, originals=(owl, moon)
)

galaxy_path = data_dir / 'downloaded' / 'galaxy.png'

cat = load_small_image(data_dir / 'downloaded' / 'cat.jpg', max_size=1600)
tiger = load_small_image(data_dir / 'downloaded' / 'tiger.jpg', max_size=960)
cat_aligned = align_by_eyes(
    cat, [(674, 686), (1014, 568)], [(450, 398), (642, 410)], tiger.shape
)
top, bottom, left, right = 150, 770, 236, 856
save_hybrid_results(
    resize_image(tiger[top:bottom, left:right], (480, 480)),
    resize_image(cat_aligned[top:bottom, left:right], (480, 480)),
    'tiger_cat', high_sigma=4, low_sigma=10, high_weight=1.2,
    already_aligned=True, save_analysis=True, originals=(tiger, cat)
)

# Part 2.3 results
apple = sk.img_as_float(skio.imread(data_dir / 'apple.jpeg'))
orange = sk.img_as_float(skio.imread(data_dir / 'orange.jpeg'))
save_stacks(apple, 'apple')
save_stacks(orange, 'orange')

# Part 2.4 results
oraple_mask = vertical_mask(apple.shape[:2])
save_blend_results(
    apple, orange, oraple_mask, 'oraple', save_process=True
)

summer = load_cropped_image(data_dir / 'downloaded' / 'crystal_summer.jpg', (640, 960))
winter = load_cropped_image(data_dir / 'downloaded' / 'crystal_winter.jpg', (640, 960))
winter = cv2.warpAffine(
    winter, np.float32([[1, 0, 0], [0, 1, -24]]), (960, 640),
    borderMode=cv2.BORDER_REFLECT_101
)
# Summer on the left, winter on the right. The mask is blurred first so the
# lake turns from open water to ice gradually instead of along a line.
season_mask = low_pass(vertical_mask(summer.shape[:2]), 60)
save_blend_results(
    summer, winter, season_mask, 'seasons'
)

# Irregular mask: the Rock's side-eye morphed onto Djokovic checking his racket.
# The 478 matching face landmarks were found once with MediaPipe FaceMesh and
# saved to data/rock_djokovic_landmarks.json. Djokovic's points are in a
# 1000 x 1000 crop of his photo.
with open(data_dir / 'rock_djokovic_landmarks.json') as f:
    landmarks = json.load(f)
left, top = landmarks['crop_origin']
djokovic = load_small_image(data_dir / 'downloaded' / 'djokovic_racket.jpg', max_size=3264)
djokovic = djokovic[top:top + 1000, left:left + 1000]
rock = load_small_image(data_dir / 'downloaded' / 'rock_side_eye.jpg', max_size=463)
rock_points = np.array(landmarks['rock'])
djokovic_points = np.array(landmarks['djokovic'])

# Morph: split the face into triangles between landmarks and warp each Rock
# triangle onto the matching Djokovic triangle.
morph = sk.transform.PiecewiseAffineTransform()
morph.estimate(djokovic_points, rock_points)
rock_aligned = sk.transform.warp(rock, morph, output_shape=(1000, 1000), mode='edge')

# Mask: Djokovic's face outline, shrunk a little toward the center, with the
# top of the forehead cut off so the Rock's bald head doesn't cover the hair.
center = djokovic_points.mean(axis=0)
outline = cv2.convexHull(np.int32(center + 0.88 * (djokovic_points - center)))
rock_mask = np.zeros((1000, 1000), np.float32)
cv2.fillConvexPoly(rock_mask, outline, 1)
rows = np.where(rock_mask.any(axis=1))[0]
rock_mask[:int(rows.min() + 0.08 * (rows.max() - rows.min()))] = 0
# Skip anything that came from the white border of the meme image.
rock_face = cv2.erode((rock.min(axis=2) < 0.8).astype(np.float32),
                      np.ones((9, 9), np.uint8))
rock_mask *= sk.transform.warp(rock_face, morph, output_shape=(1000, 1000)) > 0.5

# The Rock is paler and lit differently. Match the mean and spread of each
# color channel inside the mask to Djokovic's skin before blending.
inside = rock_mask > 0.5
for channel in range(3):
    source = rock_aligned[:, :, channel]
    target = djokovic[:, :, channel]
    rock_aligned[:, :, channel] = (
        (source - source[inside].mean()) / source[inside].std()
        * target[inside].std() + target[inside].mean())
rock_aligned = np.clip(rock_aligned, 0, 1)

rock_djokovic = save_blend_results(rock_aligned, djokovic, rock_mask, 'rock_djokovic')
hard_cut = rock_mask[:, :, np.newaxis] * rock_aligned
hard_cut += (1 - rock_mask[:, :, np.newaxis]) * djokovic
save(hard_cut[100:550, 250:700], 'rock_djokovic_face_hard_cut.jpg', 'part2_4')
save(rock_djokovic[100:550, 250:700], 'rock_djokovic_face_blend.jpg', 'part2_4')
save(rock, 'rock_djokovic_rock.jpg', 'part2_4')

# A failed irregular blend: a red rose cut out of its black background and
# placed on a spiral galaxy.
galaxy = load_cropped_image(galaxy_path, (480, 480), (90, 80, 880, 870))
rose_black = load_cropped_image(
    data_dir / 'downloaded' / 'rose_black.jpg', (480, 480), (110, 9, 830, 729)
)
rose_mask = ((rose_black[:, :, 0] > 0.12) &
             (rose_black[:, :, 0] > 1.35 * rose_black[:, :, 1]) &
             (rose_black[:, :, 0] > 1.25 * rose_black[:, :, 2])).astype(np.uint8)
rose_mask = cv2.morphologyEx(rose_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
rose_mask = cv2.morphologyEx(rose_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
save_blend_results(rose_black, galaxy, rose_mask.astype(float), 'cosmic_rose')

# Irregular mask: the spiral galaxy blended into the iris of an eye.
eye = load_small_image(data_dir / 'downloaded' / 'iris_eye.jpg', max_size=1280)
galaxy_full = load_small_image(galaxy_path, max_size=960)
height, width = eye.shape[:2]
iris_x, iris_y, iris_r = 616, 330, 195
pupil_x, pupil_y, pupil_r = 615, 358, 74

# Scale the galaxy so its core sits on the pupil and its arms fill the iris.
galaxy_transform = cv2.getRotationMatrix2D((480, 438), 0, 0.6)
galaxy_transform[0, 2] += pupil_x - 480
galaxy_transform[1, 2] += pupil_y - 438
galaxy_eye_image = np.clip(cv2.warpAffine(
    galaxy_full, galaxy_transform, (width, height),
    borderMode=cv2.BORDER_REFLECT_101), 0, 1)

# Mask: iris disc below the upper eyelid, minus the pupil and catchlight.
rows, columns = np.mgrid[:height, :width]
iris = (columns - iris_x)**2 + (rows - iris_y)**2 < iris_r**2
below_lid = rows > 238 + 0.0016 * (columns - iris_x)**2
pupil = (columns - pupil_x)**2 + (rows - pupil_y)**2 < (pupil_r + 4)**2
bright = ((eye.mean(axis=2) > 0.62) & iris).astype(np.uint8)
_, labels, _, _ = cv2.connectedComponentsWithStats(bright)
catchlight = cv2.dilate((labels == labels[265, 615]).astype(np.uint8),
                        np.ones((9, 9), np.uint8)).astype(bool)
eye_mask = (iris & below_lid & ~pupil & ~catchlight).astype(np.uint8)
eye_mask = cv2.morphologyEx(eye_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
_, labels, stats, _ = cv2.connectedComponentsWithStats(eye_mask)
eye_mask = (labels == 1 + np.argmax(stats[1:, 4])).astype(np.uint8)
eye_mask = cv2.morphologyEx(eye_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
eye_mask = eye_mask.astype(bool) & iris & below_lid & ~pupil & ~catchlight

# The galaxy's bright core would bleed into the pupil through the coarse
# levels, so cover the core with a dark disc the color of the real pupil.
pupil_color = np.median(eye[pupil & ~catchlight], axis=0)
keep = cv2.GaussianBlur(pupil.astype(float), (0, 0), 3)[:, :, np.newaxis]
galaxy_eye_image = keep * pupil_color + (1 - keep) * galaxy_eye_image

top, bottom, left, right = 70, 630, 316, 916
galaxy_eye_image = galaxy_eye_image[top:bottom, left:right]
eye = eye[top:bottom, left:right]
eye_mask = eye_mask[top:bottom, left:right].astype(float)
save_blend_results(
    galaxy_eye_image, eye, eye_mask, 'galaxy_eye', save_process=True
)
save(galaxy_full, 'galaxy_eye_galaxy.jpg', 'part2_4')

