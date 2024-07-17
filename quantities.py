import numpy as np
from numpy import conj, real
from numpy.fft import fft2, fftshift, ifft2
from scipy import ndimage
from scipy.signal import convolve2d
from scipy import stats

def ROI(array, y, x):
    y0, x0 = ndimage.center_of_mass(np.isnan(array) == 0)
    ROI_center = [int(y0), int(x0)] # y and x 
    ROI_shape = [y, x]
    # ROI center an area.Copy and paste from MADS
    ROI = (slice(ROI_center[0]-ROI_shape[0]//2, ROI_center[0]+ROI_shape[0]//2), 
           slice(ROI_center[1]-ROI_shape[1]//2, ROI_center[1]+ROI_shape[1]//2))
    return ROI

def strain(array):
    theta = array/2
    value = 100 * (1 - (np.sin(np.pi*theta/180)/np.sin(np.pi*np.nanmean(theta)/180)))
    return value

# def microstrain():
def normalize(array):
    return (array-np.nanmin(array))/(np.nanmax(array)-np.nanmin(array))

def misorientation(array1, array2):
    kernel =  np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    mean_kernel = np.sum(kernel)
    result1 = convolve2d(array1 - np.nanmean(array1), kernel/mean_kernel, mode='same', boundary='symm')
    result2 = convolve2d(array2 - np.nanmean(array2), kernel/mean_kernel, mode='same', boundary='symm')
    result = np.sqrt(result1**2 + result2**2)
    return result

def autocorrelation(data):
    #
    array = np.where(np.isnan(data), np.nanmean(data), data)
    # normalization
    array = array - np.nanmin(array)
    array = array / np.nansum(array)
    array = array - np.nanmean(array)

    # calculate correlation matrix
    crmx = real(ifft2(fft2(array) * conj(fft2(array))))
    shift= fftshift(crmx)/np.max(crmx) #peak centered
    
    return shift

def radial_profile(data, mode = None):
    # Get the dimensions of the data
    y, x = np.indices((data.shape))
    # Compute the center of the image
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    # Calculate the distance of each pixel from the center
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Flatten the radius and data arrays
    r_flat = r.flatten()
    data_flat = data.flatten()
    # Bin the data based on the radius, i.e., distance from the center
    # Use the maximum radius to define the bin edges
    max_radius = np.max(r)
    bins = np.arange(0, max_radius + 1, 1)
    # Digitize the radii to bin numbers
    bin_indices = np.digitize(r_flat, bins)
    # Use bin indices to find which pixels fall into each bin
    # Then, calculate the mean value for each bin
    if mode == 'mean':
        radial_param = [np.mean(data_flat[bin_indices == i]) for i in range(1, len(bins))]
    if mode == 'median':
        radial_param = [np.median(data_flat[bin_indices == i]) for i in range(1, len(bins))]
    if mode == 'std':
        radial_param = [np.std(data_flat[bin_indices == i]) for i in range(1, len(bins))]
    if mode == 'max':
        radial_param = [np.max(data_flat[bin_indices == i]) for i in range(1, len(bins))]
    if mode == 'min':
        radial_param = [np.min(data_flat[bin_indices == i]) for i in range(1, len(bins))]
    
    return np.array(radial_param), bins[:-1]

def grain_mean_distances(array, mode = None):
    flat_array = array.flatten()
    mask = ~np.isnan(array)
    distances = ndimage.distance_transform_edt(mask)
    max_dist_px = int(np.max(distances))
    bins_dist = np.arange(0, max_dist_px + 1, 1)
    bin_indices = np.digitize(distances.flatten(), bins_dist)
    if mode == "normal":
        radial_mean = [np.average(flat_array[bin_indices == i]) for i in range(1, len(bins_dist))]
    if mode == "energy":
        radial_mean = [np.mean(flat_array[bin_indices == i])**2 for i in range(1, len(bins_dist))]
    if mode == "std1":
        radial_mean = [np.std(flat_array[bin_indices == i]) for i in range(1, len(bins_dist))]
    if mode == "std2":
        radial_mean = [stats.sem(flat_array[bin_indices == i]) for i in range(1, len(bins_dist))]
        
    return bins_dist[:-1], np.array(radial_mean)
    


