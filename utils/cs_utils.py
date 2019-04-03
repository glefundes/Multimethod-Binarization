import cv2
import numpy as np

from scipy.optimize import curve_fit

# Plot list of ROIs to image
def plot_blobs(img, boxes, centroids=None, plot_line=False):
    if(plot_line and centroids != None):
        centroids = sorted(centroids, key=lambda x: x[0])
        # find least square line coefficients
        A, B = get_least_square_coef_from_centroids(centroids)

        pt_1 = (0,int(A*0+B))
        pt_2 = (img.shape[1],int(A*img.shape[1]+B))

        ret_img = cv2.line(img, pt_1, pt_2, (0,255,0), 1)

    for x1,y1,x2,y2 in boxes:
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)
    return img

def get_least_square_coef_from_centroids(centroids):
    A,B = curve_fit(lambda x, A, B: A*x + B, [x for x,_ in centroids], [y for _,y in centroids])[0]
    return A,B

def distance_from_line(point,coef):
    return abs((coef[0]*point[0])-point[1]+coef[1])/math.sqrt((coef[0]*coef[0])+1)

# Use CCA to extract blobs from binary source
def get_candidate_regions(bin_img):
    bin_img = np.uint8(bin_img)
    _, labels, stats, centroids =  cv2.connectedComponentsWithStats(bin_img, connectivity=4)
    # Discard uninteresting blobs
    candidate_boxes, centroids = discard_blobs(stats, bin_img.shape, centroids)

    return (candidate_boxes, centroids)

# Discard uninteresting blob candidates
def discard_blobs(cca_stats, input_shape, centroids):
    valid_blobs = []
    valid_centroids = []

    for blob_idx in range(len(cca_stats)):
        blob_h = cca_stats[blob_idx,cv2.CC_STAT_HEIGHT]
        blob_w = cca_stats[blob_idx,cv2.CC_STAT_WIDTH]
        blob_area = cca_stats[blob_idx,cv2.CC_STAT_AREA]
        input_w = input_shape[1]
        input_h = input_shape[0]
        cx, cy = centroids[blob_idx]
        # discard by size
        if blob_w < 0.05*input_w : continue
        if blob_w > 0.125*input_w : continue
        if blob_h < 0.2*input_h : continue
        if blob_h > 0.45*input_h : continue

        # discard by area
        if blob_area < (0.1*blob_h*blob_w): continue

        #discard by position
        blob_center_y = cca_stats[blob_idx,cv2.CC_STAT_TOP]+ (blob_h/2)

        if blob_center_y < 0.15*input_h: continue
        if blob_center_y > 0.85*input_h: continue

        # discard by boundaries
        if cx < 0.015*input_w and cy < 0.015*input_h: continue
        if cx < 0.015*input_w and cy > 0.985*input_h: continue
        if cx > 0.985*input_w and cy < 0.015*input_h: continue
        if cx > 0.985*input_w and cy > 0.985*input_h: continue



        x1, y1 = cca_stats[blob_idx,cv2.CC_STAT_LEFT], cca_stats[blob_idx,cv2.CC_STAT_TOP]
        x2 = x1 + cca_stats[blob_idx,cv2.CC_STAT_WIDTH]
        y2 = y1 + cca_stats[blob_idx,cv2.CC_STAT_HEIGHT]
        valid_blobs.append([x1,y1,x2,y2])
        valid_centroids.append((cx,cy))

    return valid_blobs, valid_centroids
