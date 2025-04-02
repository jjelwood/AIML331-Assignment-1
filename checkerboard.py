import numpy as np
import matplotlib.pyplot as plt
import math

# Load the image
img_path = 'checkerboard.png'

image = plt.imread(img_path)

# Convert to grayscale if the image is in color
if image.ndim == 3:
    image = np.mean(image, axis=2)

# Convert image to [0,255] range
if img_path.endswith('.png'):
    image = image * 255

assert image.shape[0] == image.shape[1], "Image must be square"
assert np.all(0 <= image) and np.all(image <= 255), "Image must be in [0,255] range"

def get_counts(image):
    image = np.round(image).astype(int)
    counts = np.zeros((np.max(image)+1,), dtype=int)
    for i in range(len(counts)):
        counts[i] = np.sum(image == i)
    return counts

def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

def cumulative_dist(image):
    counts = get_counts(image)
    cumulative_histogram = np.zeros((len(counts),), dtype=float)
    cumulative_histogram[0] = counts[0]
    for i in range(1, len(counts)):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + counts[i]
    return cumulative_histogram / np.sum(counts)

def convolution(image, kernel):
    kernel = kernel[::-1, ::-1] # flip the kernel
    kernel_size = kernel.shape[0]
    pad_width = math.ceil(kernel_size / 2)
    padded_image = np.pad(image, pad_width)
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output

def box_filter(kernel_size):
    return np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)

def plt_histograms(name, image):
    plt.figure(figsize=(15, 5))

    # Plot the image
    plt.subplot(1, 3, 1)
    show_image(image)
    plt.title(name)
    plt.axis('off')

    # Plot the histogram
    plt.subplot(1, 3, 2)
    histogram = get_counts(image) / np.sum(get_counts(image))
    plt.bar(list(range(len(histogram))), histogram, width=1, color='blue')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('PDF Histogram')

    # Plot the cumulative histogram
    plt.subplot(1, 3, 3)
    cumulative_histogram = cumulative_dist(image)
    plt.bar(list(range(len(cumulative_histogram))), cumulative_histogram, width=1, color='blue')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF Histogram')

    # Plot the FFT of the image on the second row
    # fft_image = np.fft.fftshift(np.fft.fft2(image))

    # plt.subplot(2, 3, 4)
    # # Better visualization of the FFT
    # amplitude = np.log(np.abs(fft_image) + 1)
    # plt.imshow(amplitude, cmap='gray')
    # plt.title('Amplitude of FFT')

    # plt.subplot(2, 3, 5)
    # angle = np.angle(fft_image)
    # plt.imshow(angle, cmap='gray', vmin=-np.pi, vmax=np.pi)
    # plt.title('Phase of FFT')

    plt.tight_layout()
    plt.show()

def fft_convolution(image, kernel):
    kernel_size = kernel.shape[0]
    image_size = image.shape[0]
    image_pad = kernel_size // 2
    pad_width = image_size - kernel_size
    padded_kernel = np.pad(kernel, ((pad_width + image_pad, 0), (pad_width + image_pad, 0)))
    padded_image = np.pad(image, ((image_pad, 0), (image_pad, 0)))
    kernel_fft = np.fft.fft2(padded_kernel)
    image_fft = np.fft.fft2(padded_image)
    return np.fft.ifft2(image_fft * kernel_fft).real[image_pad:image_size-image_pad, image_pad:image_size-image_pad]

def histogram_equalization(image):
    cumulative_histogram = cumulative_dist(image)
    equalized_image = np.vectorize(lambda j: 255 * cumulative_histogram[int(j)])(image)

    return equalized_image

filter_kernel = box_filter(30)

plt_histograms("Base Image", image)
plt_histograms("30 x 30 Box Blur", convolution(image, filter_kernel))
plt_histograms("30 x 30 Box Blur, FFT", fft_convolution(image, filter_kernel))
plt_histograms("Equalized Image", histogram_equalization(convolution(image, filter_kernel)))