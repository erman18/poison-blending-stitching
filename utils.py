
import numpy as np
from matplotlib import pyplot as plt


import os
import re

# Read files in a folder.
def readFiles(folder_path, filter="mask"):
    # Get all files in the folder.
    files = os.listdir(folder_path)
    
    # Get the file paths.
    file_paths = [os.path.join(folder_path, file) for file in files if re.match(rf"^{filter}.*\.png$", file, re.I)]
    
    # Sort the file paths.
    file_paths.sort()
    print(f"Found {len(file_paths)} files in {folder_path}")
    print(file_paths)
    return file_paths

# Divide by a by b and replace undefined values by `fill`.
def divAB( a, b, fill=np.nan ):
    """ a / b, divide by 0 -> `fill`
        div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
        div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
    if np.isscalar( c ):
        return c if np.isfinite( c ) else fill
    else:
        c[ ~ np.isfinite( c )] = fill
        return c

def printRandom3DSample(Arr, numElts=(5, 2, 3)):
    # A is a 3d-array
    np.random.seed(30)
    idx = np.random.randint(Arr.shape[0], size=numElts[0])
    idy = np.random.randint(Arr.shape[1], size=numElts[1])
    idz = np.random.randint(Arr.shape[2], size=numElts[2])
    idxyz = [(x, y, z) for x in idx for y in idy for z in idz]
    data = [Arr[p] for p in idxyz]
    print("idx: ", idx)
    print("idy: ", idy)
    print("idz: ", idz)
    # print("idxyz: ", idxyz)
    print("data: ", data)

def generate_histogram(img):
    """
    @params: img: is a grayscale. We calculate the Normalized histogram of this image.
    @params: do_print: if or not print the result histogram
    @return: will return both histogram and the grayscale image 
    """
    if len(img.shape) == 3: # img is colorful, so we convert it to grayscale
        gr_img = np.mean(img, axis=-1)
    else:
        gr_img = img

    '''now we calc grayscale histogram'''
    gr_hist = np.zeros([256])
    for x_pixel in range(gr_img.shape[0]):
        for y_pixel in range(gr_img.shape[1]):
            pixel_value = int(gr_img[x_pixel, y_pixel])
            gr_hist[pixel_value] += 1
            
    '''normalizing the Histogram'''
    gr_hist /= (gr_img.shape[0] * gr_img.shape[1] - gr_hist[0])
    gr_hist[0] = 0
    return gr_hist

def generate_combined_histogram(imgs):
    """
    This take a list of rgb images and split into simple images
    then generate the list of all the images and merge all them all
    together to return a three histogram for each channel
    @params: imgs: is a list of rgb images.
    @return: hr, hg, hb: the combined histogram for each rgb channel
    """
    hb = np.zeros([256])
    hg = np.zeros([256])
    hr = np.zeros([256])
    
    K = len(imgs)
    for k in range(K):
        '''Retrieved the channel'''
        b = imgs[k][:,:,0]
        g = imgs[k][:,:,1]
        r = imgs[k][:,:,2]
        
        '''Generate the histogram of each channels'''
        _hb = generate_histogram(b)
        _hg = generate_histogram(g)
        _hr = generate_histogram(r)
        
        '''Combined histogram'''
        hb += (1/K)*_hb
        hg += (1/K)*_hg
        hr += (1/K)*_hr

    return hb, hg, hr

def apply_histogram(img, histo, L=256):
    """This function apply a combined histogram to the input image.
    @param: img: RGB channel
    @param: histo: (hb, hg, hr) the 3-channel histogram.
    @return: return the enhanced image
    """
    hb, hg, hr = histo    
    en_img = np.zeros_like(img)
    
    eq_hb = np.zeros_like(hb)
    eq_hg = np.zeros_like(hg)
    eq_hr = np.zeros_like(hr)
    for i in range(len(hb)):
        eq_hb[i] = np.floor((L - 1) * np.sum(hb[0:i]))
        eq_hg[i] = np.floor((L - 1) * np.sum(hg[0:i]))
        eq_hr[i] = np.floor((L - 1) * np.sum(hr[0:i]))
    
    '''enhance image as well:'''
    for x_pixel in range(img.shape[0]):
        for y_pixel in range(img.shape[1]):
            pixel_val_b = int(img[x_pixel, y_pixel, 0])
            pixel_val_g = int(img[x_pixel, y_pixel, 1])
            pixel_val_r = int(img[x_pixel, y_pixel, 2])
            en_img[x_pixel, y_pixel, 0] = eq_hb[pixel_val_b]
            en_img[x_pixel, y_pixel, 1] = eq_hg[pixel_val_g]
            en_img[x_pixel, y_pixel, 2] = eq_hr[pixel_val_r]

    return en_img


def plot_histogram(_histrogram, title, name="", block=False):
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='#ef476f')
    plt.bar(np.arange(len(_histrogram)), _histrogram, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.show()

def print_histogram(_histrogram, title, name=""):
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='#ef476f')
    plt.bar(np.arange(len(_histrogram)), _histrogram, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.show()
    plt.savefig("hist_" + name)


#def equalize_histogram(img, histo, L):
#    eq_histo = np.zeros_like(histo)
#    en_img = np.zeros_like(img)
#    for i in range(len(histo)):
#        eq_histo[i] = int((L - 1) * np.sum(histo[0:i]))
#    print_histogram(eq_histo, title="Equalized Histogram", name="eq_"+str(index))
#    '''enhance image as well:'''
#    for x_pixel in range(img.shape[0]):
#        for y_pixel in range(img.shape[1]):
#            pixel_val = int(img[x_pixel, y_pixel])
#            en_img[x_pixel, y_pixel] = eq_histo[pixel_val]
#    '''creating new histogram'''
#    hist_img, _ = generate_histogram(en_img, print=False, index=index)
#    print_img(img=en_img, histo_new=hist_img, histo_old=histo, index=str(index), L=L)
#    return eq_histo

def print_img(img, histo_new, histo_old, index, L):
    dpi = 80
    width = img.shape[0]
    height = img.shape[1]
    if height > width:
        figsize = (img.shape[0]*4) / float(dpi), (height)/ float(dpi)
        fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 1, 1]}, figsize=figsize)
    else:
        figsize = (width) / float(dpi), (height*4) / float(dpi)
        fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=figsize)

    fig.suptitle("Enhanced Image with L:" + str(L))
    axs[0].title.set_text("Enhanced Image")
    axs[0].imshow(img, vmin=np.amin(img), vmax=np.amax(img), cmap='gray')

    axs[1].title.set_text("Equalized histogram")
    axs[1].plot(histo_new, color='#f77f00')
    axs[1].bar(np.arange(len(histo_new)), histo_new, color='#003049')

    axs[2].title.set_text("Main histogram")
    axs[2].plot(histo_old, color='#ef476f')
    axs[2].bar(np.arange(len(histo_old)), histo_old, color='#b7b7a4')
    plt.tight_layout()
    plt.savefig("e" + index + str(L)+".pdf")
    plt.savefig("e" + index + str(L)+".png")

# def find_value_target(val, target_arr):
    # key = np.where(target_arr == val)[0]

    # if len(key) == 0:
        # key = find_value_target(val+1, target_arr)
        # if len(key) == 0:
            # key = find_value_target(val-1, target_arr)
    # vvv = key[0]
    # return vvv

def find_value_target(val, target_arr):
    idx = (np.abs(target_arr - val)).argmin()
    return target_arr[idx]


#def match_histogram(inp_img, hist_input, e_hist_input, e_hist_target, _print=True):
#    '''map from e_inp_hist to 'target_hist '''
#    en_img = np.zeros_like(inp_img)
#    tran_hist = np.zeros_like(e_hist_input)
#    for i in range(len(e_hist_input)):
#        tran_hist[i] = find_value_target(val=e_hist_input[i], target_arr=e_hist_target)
#    print_histogram(tran_hist, title="Transferred Histogram", name="trans_hist_")
#    '''enhance image as well:'''
#    for x_pixel in range(inp_img.shape[0]):
#        for y_pixel in range(inp_img.shape[1]):
#            pixel_val = int(inp_img[x_pixel, y_pixel])
#            en_img[x_pixel, y_pixel] = tran_hist[pixel_val]
#    '''creating new histogram'''
#    hist_img, _ = generate_histogram(en_img, print=False, index=3)
#    print_img(img=en_img, histo_new=hist_img, histo_old=hist_input, index=str(3), L=L)

def make_data_3D(x_range, y_range, step):
     X = np.arange(-x_range, x_range, step)
     Y = np.arange(-y_range, y_range, step)
     X, Y = np.meshgrid(X, Y)
     R = np.sqrt(X**2 + Y**2)
     Z = np.sin(R)
     return X, Y, Z

