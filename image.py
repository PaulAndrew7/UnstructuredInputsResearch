import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calculate_mse(original, processed):
    """Calculate Mean Squared Error between two images"""
    return np.mean((original - processed) ** 2)

def calculate_psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    mse = calculate_mse(original, processed)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def apply_frequency_enhancement(image, radius=20):
    """Apply frequency domain enhancement optimized for document images"""
    # Convert to float for better precision
    img_float = image.astype(np.float32)
    
    # Apply FFT
    dft = np.fft.fft2(img_float)
    dft_shifted = np.fft.fftshift(dft)
    
    rows, cols = img_float.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create frequency domain filter (high-pass for text sharpening)
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    d = np.sqrt(x*x + y*y)
    
    # Gaussian high-pass filter (smaller radius for documents)
    gaussian_highpass = 1 - np.exp(-((d**2) / (2.0 * (radius/100)**2)))
    
    # Apply filter
    filtered_dft_shifted = dft_shifted * gaussian_highpass
    filtered_dft = np.fft.ifftshift(filtered_dft_shifted)
    img_back = np.fft.ifft2(filtered_dft)
    img_back = np.abs(img_back)
    
    # Normalize and convert back to uint8
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

def apply_color_correction_to_gray(image):
    """Apply color correction principles to grayscale images"""
    # Convert to LAB-like processing for grayscale
    # Normalize the image to have better contrast
    mean_val = np.mean(image)
    std_val = np.std(image)
    
    # Adjust contrast and brightness
    corrected = image.astype(np.float32)
    corrected = (corrected - mean_val) * (50 / std_val) + 128
    corrected = np.clip(corrected, 0, 255)
    
    return corrected.astype(np.uint8)

def apply_unsharp_mask(image, sigma=1.5, alpha=1.8):
    """Apply unsharp masking for text enhancement"""
    # Create Gaussian blur
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # Apply unsharp mask
    unsharp_mask = cv2.addWeighted(image, alpha, gaussian_blur, -(alpha-1), 0)
    return unsharp_mask

def enhanced_preprocess_document(image_path):
    """
    Enhanced document preprocessing - just frequency enhancement and inversion
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_gray = gray.copy()
    
    # Step 1: Apply frequency domain enhancement
    freq_enhanced = apply_frequency_enhancement(gray, radius=15)
    
    # Step 2: Invert the image
    inverted = 255 - freq_enhanced
    
    return {
        'original': original_gray,
        'frequency_enhanced': freq_enhanced,
        'final': inverted
    }

def compare_preprocessing_methods(image_path):
    """
    Compare original vs enhanced preprocessing methods
    """
    # Original method
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_orig = cv2.fastNlMeansDenoising(gray)
    thresh_orig = cv2.adaptiveThreshold(
        denoised_orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((1, 1), np.uint8)
    original_result = cv2.morphologyEx(thresh_orig, cv2.MORPH_CLOSE, kernel)
    
    # Enhanced method
    enhanced_results = enhanced_preprocess_document(image_path)
    enhanced_result = enhanced_results['final']
    
    # Calculate metrics
    mse_value = calculate_mse(gray, enhanced_result)
    psnr_value = calculate_psnr(gray, enhanced_result)
    ssim_value = ssim(gray, enhanced_result)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    # Original grayscale
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    # Original preprocessing result
    plt.subplot(2, 3, 2)
    plt.imshow(original_result, cmap='gray')
    plt.title('Original Preprocessing')
    plt.axis('off')
    
    # Frequency enhanced
    plt.subplot(2, 3, 3)
    plt.imshow(enhanced_results['frequency_enhanced'], cmap='gray')
    plt.title('Frequency Enhanced')
    plt.axis('off')
    
    # Final enhanced result (inverted)
    plt.subplot(2, 3, 4)
    plt.imshow(enhanced_result, cmap='gray')
    plt.title(f'Enhanced Final (Inverted)\nMSE: {mse_value:.2f}\nPSNR: {psnr_value:.2f}\nSSIM: {ssim_value:.3f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original_result': original_result,
        'enhanced_result': enhanced_result,
        'metrics': {
            'mse': mse_value,
            'psnr': psnr_value,
            'ssim': ssim_value
        }
    }

# Example usage:
# results = compare_preprocessing_methods('path_to_your_document_image.jpg')
# 
# # Or just get the enhanced preprocessing result:
# enhanced_doc = enhanced_preprocess_document('path_to_your_document_image.jpg')
# final_image = enhanced_doc['final']
#Example usage:
results = compare_preprocessing_methods("images\prescription2.png")

# # Or just get the enhanced preprocessing result:
# enhanced_doc = enhanced_preprocess_document('path_to_your_document_image.jpg')
# final_image = enhanced_doc['final']