import cv2
import numpy as np
import bottleneck as bn

def get_nilblack_threshold(m, s, k):
        return m + k*s

def get_sauvola_threshold(m, s, k):
    R = np.amax(s)
    return m*(1 + k*(s/(R-1)))

def get_wolf_threshold(m,s,k,M):
    R = np.amax(s)
    return m + k*(s/(R-1))*(m-M)

def binarize(img, methods, resize=None, return_original=False, morph_kernel=None):
    results = []
    # Resize image to desired dimensions
    if (resize):
        img = cv2.resize(img, dsize=resize, interpolation=cv2.INTER_CUBIC)
    # Save resized original image for visualization purposes. Will be returned on first position of result array
    if(return_original):
        results.append(img)
    # Convert input to Grayscale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Normalize image pixels to [0..1] values
    img = img/255

    for method in methods:
        binary_mask = np.zeros(img.shape)

        w = method['window_size']
        k = method['k_factor']

        # Moving window average and standard deviation
        m = bn.move_mean(img, window=int(w), min_count=1)
        s = bn.move_std(img, window=int(w), min_count=1)

        if (method['type'] == 'niblack'):
            T = get_nilblack_threshold(m,s,k)

        elif (method['type'] == 'sauvola'):
            T = get_sauvola_threshold(m,s,k)

        elif (method['type'] == 'wolf'):
            M = cv2.minMaxLoc(img)[0]
            T = get_wolf_threshold(m,s,k,M)

        else:
            raise("invalid method passed! try using 'niblack', 'sauvola' or 'wolf'.")

        binary_mask = np.where(img<T, 1.0, 0.0)

        try:
            # Use morphological opening to improve binary image quality
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        except:
            # No morphology kernel defined
            pass
        results.append(binary_mask)

    return results
