import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure

from skimage.morphology import opening
from skimage.morphology import disk

import fabio

def grain_strain(file, cmap):
    now = h5py.File(file, "r")
    diffry_kurt = (np.array(now["entry/diffry/Kurtosis/Kurtosis"]))
    obpitch_kurt = (np.array(now["entry/obpitch/Kurtosis/Kurtosis"]))

    com_obpitch = (np.array(now["entry/obpitch/Center of mass/Center of mass"]))
    com_diffry = (np.array(now["entry/diffry/Center of mass/Center of mass"]))

    obpitch_FWHM = (np.array(now["entry/obpitch/FWHM/FWHM"]))
    diffry_FWHM = (np.array(now["entry/diffry/FWHM/FWHM"]))

    obpitch_skew = (np.array(now["entry/obpitch/Skewness/Skewness"]))
    diffry_skew = (np.array(now["entry/diffry/Skewness/Skewness"]))

    diffry_maps = [com_diffry, diffry_FWHM, diffry_skew, diffry_kurt]
    obpitch_maps = [com_obpitch, obpitch_FWHM, obpitch_skew, obpitch_kurt]
    
    mosaicity = (np.array(now["entry/Mosaicity/Mosaicity"]))
    
    
    fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (12,5), dpi = 150)

    cmap = cmap
    shrink = 1
    
    cd = axs[0,0].imshow(com_diffry, cmap = cmap)
    fig.colorbar(cd, ax = axs[0,0], shrink = shrink)
    axs[0,0].set_title("COM")

    cc = axs[1,0].imshow(com_obpitch, cmap = cmap)
    fig.colorbar(cc, ax = axs[1,0], shrink = shrink)

    df = axs[0,1].imshow(diffry_FWHM, cmap = cmap)
    fig.colorbar(df, ax = axs[0,1], shrink = shrink)
    axs[0,1].set_title("FWHM")

    cf = axs[1,1].imshow(obpitch_FWHM, cmap = cmap)
    fig.colorbar(cf, ax = axs[1,1], shrink = shrink)

    ds = axs[0,2].imshow(diffry_skew, cmap = cmap)
    fig.colorbar(ds, ax = axs[0,2], shrink = shrink)
    axs[0,2].set_title("Skewness")

    cs = axs[1,2].imshow(obpitch_skew, cmap = cmap)
    fig.colorbar(cs, ax = axs[1,2], shrink = shrink)

    dk = axs[0,3].imshow(diffry_kurt, cmap = cmap)
    fig.colorbar(dk, ax = axs[0,3], shrink = shrink)
    axs[0,3].set_title("Kurtosis")

    ck = axs[1,3].imshow(obpitch_kurt, cmap = cmap)
    fig.colorbar(ck, ax = axs[1,3], shrink = shrink)

    for i in range(2):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['bottom'].set_visible(False)
            axs[i,j].spines['left'].set_visible(False)

    axs[0,0].set_ylabel("diffry", fontsize = 12)
    axs[1,0].set_ylabel("obpitch", fontsize = 12)

    plt.suptitle(file + file, y= 1, x = 0.44, weight = "bold")

    plt.tight_layout()

    plt.show()
    
    fig, ax = plt.subplots(1, 2, figsize= (12,4), dpi = 100)
    ax[0].imshow(mosaicity)
    keys = now["entry/Orientation distribution/curves"].keys()
    for key in keys:
        colors = np.array(now[f"entry/Orientation distribution/curves/{key}/color"])/255 #max number
        # RGBA scale = four numers to set a color
        points = np.array(now[f"entry/Orientation distribution/curves/{key}/points/"]) #f is important
        poly = points.T
        #
        ax[1].plot(poly[:,0],poly[:,1], color = colors, linewidth = 1.4) 
        ax[1].set_xlabel(r"2$\theta$ (deg.)")
        ax[1].set_ylabel("diffry (deg.)")
    bck = now["entry/Orientation distribution/key/image"]
    extent = (ax[1].get_xlim()[0], ax[1].get_xlim()[1], ax[1].get_ylim()[0], ax[1].get_ylim()[1])
    ax[1].imshow(bck, extent = extent, origin = "lower", aspect = "auto")
    plt.show()
    now.close()
    
    now.close()
    return diffry_maps, obpitch_maps, mosaicity
######
def grain_mosa(file, cmap):
    now = h5py.File(file, "r")
    diffry_kurt = (np.array(now["entry/diffry/Kurtosis/Kurtosis"]))
    chi_kurt = (np.array(now["entry/chi/Kurtosis/Kurtosis"]))

    com_chi = (np.array(now["entry/chi/Center of mass/Center of mass"]))
    com_diffry = (np.array(now["entry/diffry/Center of mass/Center of mass"]))

    chi_FWHM = (np.array(now["entry/chi/FWHM/FWHM"]))
    diffry_FWHM = (np.array(now["entry/diffry/FWHM/FWHM"]))

    chi_skew = (np.array(now["entry/chi/Skewness/Skewness"]))
    diffry_skew = (np.array(now["entry/diffry/Skewness/Skewness"]))
    
    diffry_maps = [com_diffry, diffry_FWHM, diffry_skew, diffry_kurt]
    chi_maps = [com_chi, chi_FWHM, chi_skew, chi_kurt]
    
    mosaicity = (np.array(now["entry/Mosaicity/Mosaicity"]))
    
    fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (12,5), dpi = 150)

    cmap = cmap
    shrink = 1
    
    cd = axs[0,0].imshow(com_diffry, cmap = cmap)
    fig.colorbar(cd, ax = axs[0,0], shrink = shrink)
    axs[0,0].set_title("COM")

    cc = axs[1,0].imshow(com_chi, cmap = cmap)
    fig.colorbar(cc, ax = axs[1,0], shrink = shrink)

    df = axs[0,1].imshow(diffry_FWHM, cmap = cmap)
    fig.colorbar(df, ax = axs[0,1], shrink = shrink)
    axs[0,1].set_title("FWHM")

    cf = axs[1,1].imshow(chi_FWHM, cmap = cmap)
    fig.colorbar(cf, ax = axs[1,1], shrink = shrink)

    ds = axs[0,2].imshow(diffry_skew, cmap = cmap)
    fig.colorbar(ds, ax = axs[0,2], shrink = shrink)
    axs[0,2].set_title("Skewness")

    cs = axs[1,2].imshow(chi_skew, cmap = cmap)
    fig.colorbar(cs, ax = axs[1,2], shrink = shrink)

    dk = axs[0,3].imshow(diffry_kurt, cmap = cmap)
    fig.colorbar(dk, ax = axs[0,3], shrink = shrink)
    axs[0,3].set_title("Kurtosis")

    ck = axs[1,3].imshow(chi_kurt, cmap = cmap)
    fig.colorbar(ck, ax = axs[1,3], shrink = shrink)

    for i in range(2):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['bottom'].set_visible(False)
            axs[i,j].spines['left'].set_visible(False)

    axs[0,0].set_ylabel("diffry", fontsize = 12)
    axs[1,0].set_ylabel("chi", fontsize = 12)

    plt.suptitle(file, y= 1, x = 0.44, weight = "bold")

    plt.tight_layout()

    plt.show()
    
    fig, ax = plt.subplots(1, 2, figsize= (12,4), dpi = 100)
    ax[0].imshow(mosaicity)
    keys = now["entry/Orientation distribution/curves"].keys()
    for key in keys:
        colors = np.array(now[f"entry/Orientation distribution/curves/{key}/color"])/255 #max number
        # RGBA scale = four numers to set a color
        points = np.array(now[f"entry/Orientation distribution/curves/{key}/points/"]) #f is important
        poly = points.T
        #
        ax[1].plot(poly[:,0],poly[:,1], color = colors, linewidth = 1.4) 
        ax[1].set_xlabel("chi (deg.)")
        ax[1].set_ylabel("diffry (deg.)")
    bck = now["entry/Orientation distribution/key/image"]
    extent = (ax[1].get_xlim()[0], ax[1].get_xlim()[1], ax[1].get_ylim()[0], ax[1].get_ylim()[1])
    ax[1].imshow(bck, extent = extent, origin = "lower", aspect = "auto")
    plt.show()
    now.close()
   
    
    return diffry_maps, chi_maps

def get_regions_larger_than(boolean_mask, size_threshold):
    '''
    takes a boolean mask and returns a new mask, discarding regions smaller than size_threshold
    input:
        boolean_mask: (n,m) numpy array of dtype bool
        size_threshold: float
    return:
        updated mask, i.e. (n,m) numpy array of dtype bool
    '''
    boolean_mask = boolean_mask.astype(bool)
    ret, thresh = cv2.threshold(255*boolean_mask.astype(np.uint8), 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find and fill relevant contour
    contour_img = np.zeros(boolean_mask.shape, dtype = np.uint8)
    for contour in contours:
        x = contour[:,0,0]
        y = contour[:,0,1]
        #plt.plot(x,y)
        area = cv2.contourArea(contour)
        if area>size_threshold:
            cv2.drawContours(contour_img, [contour], -1, color=(255), thickness=cv2.FILLED)
    return contour_img[:,:].astype(bool)
   
# def masking(array, lim = None, area = None):
#     #copy_array = np.copy(array)
#     mask = np.ones(array.shape, dtype = bool)
#     grad = np.gradient(array)
#     mod = np.sqrt(grad[0]**2 + grad[1]**2)
    
#     limit = np.percentile(mod, lim)
#     mask[mod > limit] = 1
#     mask[mod <= limit] = 0
#     new_mask =  get_regions_larger_than(mask, area)
    
#     fig, ax = plt.subplots(1, 4, figsize = (10, 5), dpi = 300)
#     cmap = "Spectral_r"
#     im0 = ax[0].imshow(array, cmap = cmap)
#     ax[0].set_title("Raw")
#     im1 = ax[1].imshow(mod, cmap = cmap)
#     ax[1].set_title("Gradient modulus")
#     im2 = ax[2].imshow(mask, cmap = cmap)
#     ax[2].set_title("Mask")
#     im3 = ax[3].imshow(new_mask)
#     ax[3].set_title("Mask by area")

#     fig.colorbar(im0, ax = ax[0], orientation = "horizontal")
#     fig.colorbar(im1, ax = ax[1], orientation = "horizontal")
#     fig.colorbar(im2, ax = ax[2], orientation = "horizontal")
#     fig.colorbar(im3, ax = ax[3], orientation = "horizontal")
#     plt.tight_layout()
#     plt.show()
#     return new_mask

def mask(array, size = None, area = None):
    mask = np.ones(array.shape, dtype = bool)
    footprint = disk(size)
    dilation_erosion = opening(array, footprint)
    
    #binarizarion
    mask[dilation_erosion >= np.max(dilation_erosion)] = 0
    mask[dilation_erosion <= np.min(dilation_erosion)] = 0
    
    new_mask = get_regions_larger_than(mask, area)
    fig, ax = plt.subplots(1, 4, dpi = 200)
    ax[0].imshow(array, cmap = "turbo")
    ax[1].imshow(dilation_erosion, cmap = "turbo")
    ax[2].imshow(mask)
    ax[3].imshow(new_mask)
    
    
    for i in range(3):
        ax[i].axis("off")
    
    plt.show()
    return new_mask
    

def apply_mask(array, new_mask):
    array_masked = []
    
    for i, data in enumerate(array):
        copies = np.copy(data)
        copies[new_mask == False] = np.nan
        array_masked.append(copies)

    return array_masked
   

def border(array):
    contours = measure.find_contours(array, 0.5, fully_connected='high')
    perimeter = np.zeros_like(array, dtype=bool)
    
    for contour in contours:
        perimeter[np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)] = True
    
    return perimeter

def edf_params(file_edf0):
    now = fabio.open(file_edf0) #only one is needed for pixel calculation
    motor_names = now.header['motor_mne'].split(' ')
    motor_pos = now.header['motor_pos'].split(' ')
    motor_pos = [float(val) for val in motor_pos] 
    motors = dict(zip(motor_names, motor_pos))
    ###
    if int(motors['ffsel']) == 0: # 0 for 2x and 60 for 10x
        optical_pixel_size = 3.75
    else:
        optical_pixel_size = 0.75
    d1 = motors['obx'] / np.cos(np.radians(motors['obpitch']))
    d2 = -motors['mainx'] / np.cos(np.radians(motors['obpitch']))-d1 # mainx is negative for some reason
    M = d2 / d1
    pixel_size = np.abs(optical_pixel_size/M)
    print("the pixel size is:", pixel_size)

    diffry_id_min = float(now.header["scan"].split()[2])
    diffry_id_max = float(now.header["scan"].split()[3])
    diffry_id_steps = int(now.header["scan"].split()[4])
    #
    chi_id_min = float(now.header["scan"].split()[-4])
    chi_id_max = float(now.header["scan"].split()[-3])
    chi_id_steps = int(now.header["scan"].split()[-2]) + 1 # Chi positions are "inclusive"
    #
    diffry_id_grid = np.linspace(diffry_id_min, diffry_id_max, diffry_id_steps)
    chi_id_grid = np.linspace(chi_id_min, chi_id_max, chi_id_steps)
    diffry_id, chi_id  = np.meshgrid(diffry_id_grid, chi_id_grid, indexing='xy')
    print(now.header["scan"])
    now.close()
    diffry_edf = [diffry_id_min, diffry_id_max, diffry_id_steps]
    chi_edf = [chi_id_min, chi_id_max, chi_id_steps]
    
    return diffry_edf, chi_edf, pixel_size

def pixel_size(file_edf0):
    now = fabio.open(file_edf0) #only one is needed for pixel calculation
    motor_names = now.header['motor_mne'].split(' ')
    motor_pos = now.header['motor_pos'].split(' ')
    motor_pos = [float(val) for val in motor_pos] 
    motors = dict(zip(motor_names, motor_pos))
    ###
    if int(motors['ffsel']) == 0: # 0 for 2x and 60 for 10x
        optical_pixel_size = 3.75
    else:
        optical_pixel_size = 0.75
    d1 = motors['obx'] / np.cos(np.radians(motors['obpitch']))
    d2 = -motors['mainx'] / np.cos(np.radians(motors['obpitch']))-d1 # mainx is negative for some reason
    M = d2 / d1
    pixel_size = np.abs(optical_pixel_size/M)
    print("the pixel size is:", pixel_size)

    now.close()
    return pixel_size
    