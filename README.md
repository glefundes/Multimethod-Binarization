# Multimethod Binarization
### Demo: [Jupyter Notebook](https://github.com/glefundes/Multimethod-Binarization/blob/master/Demo.ipynb)
### Efficient implementation of local thresholding binarization methods on Python
#### Inspired by the paper [Best Combination of Binarization Methods for License Plate Character Segmentation](https://www.researchgate.net/publication/269682463_Best_Combination_of_Binarization_Methods_for_License_Plate_Character_Segmentation)

This was originally developped while studying character segmentation methods for LPR(*License Plate Recognition*). 
The more popular global thresholding methods for binarization (Otsu, etc) are not very well suited for LPR systems, as explored by the authors on the cited paper.

In the article the authors posit that using a single binarization method with static parameters while efficient for certain conditions, is not the best approach. Since different methods or parameters will result perform best for different visual features across images or even in a single image, it follows that applying multiple methods and merging the results should yield better results.

Motivated by the lack of support for local thresholding binarization by popular computer vision libraries, I wrote this code to provide a simple interface for the use of multiple binary images in character segmentation (especially in LPR systems, but should be useful to OCR applications in general). It currently supports Niblack's, Sauvola's and Wolf's binarization methods. 
___
## Requirements
* Python3
* numpy
* OpenCV
* scipy
* bottleneck 

## Usage
#### Multimethod Binarization
The main binarization function is located in `multibin.py`. it can be used as follows:
```
import multibin as mb

img = cv2.imread(img_path)
bin_imgs = mb.binarize(img, bin_methods)
```
Optional arguments are:
* **resize**: Resize input image to desired output dimensions;
* **morph_kernel**: Define a morphological kernel to be used in opening image to reduce noise (see: [this line](https://github.com/glefundes/Multimethod-Binarization/blob/25577bf57945f70ecd1e17080664952ace0fc2b6/multibin.py#L55));
* **return_original**: Return a copy of the original image resized to output dimensions as the first position in the resulting array;

You can find an example using all of them on the [demo notebook included in this repo](https://github.com/glefundes/Multimethod-Binarization/blob/master/Demo.ipynb).

This function returns a list containing one binary image for method described. The methods are defined as a list dictionary objects with the following format:
```
    {
    'type' : Binarization method (string),
    'window_size': Moving square window dimension (int),
    'k_factor': Constant (int)
    }
```
You can read more about the window size an k constant selection on the paper that inspired this code. The threshold is calculated using [bottleneck](https://github.com/kwgoodman/bottleneck) internally to speed up obtaining the moving average and standar deviation parameters.

#### CCA Analysis for ROI selection
Some auxiliary functions are defined in `utils/cs_utils.py` that serve to demonstrate how to select potential ROIs from a binarized image. This is very rundimentary since I've given up on using this method on my original project and moved on to Deep Learning methods instead. Anyway, should anyone ever need or want to explore binarization-based OCR it should be a helpful start. The [demonstration notebook](https://github.com/glefundes/Multimethod-Binarization/blob/master/Demo.ipynb) should be useful in visualizing what the system is doing.

Features: 
- [x] Wolf's, Sauvola's and Niblack's local thresholding methods
- [x] CCA analysis for blob extraction
- [x] Discard uninsteresting blobs following guidelines from [this work](https://ieeexplore.ieee.org/document/6084002/)
- [ ] Other local thresholding algorithms as described in the paper 
- [ ] Perform non-maximum supression on redundant regions
- [ ] Implement character recognition for final ROIs

I'll probably not be coming back to work on this anymore, but should anyone feel the urge to continue the work, I'll happily be of assistance.
