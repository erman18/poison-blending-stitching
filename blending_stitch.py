
import cv2 as cv
import numpy as np
import scipy.io as sio
import time, gc, sys
from matplotlib import pyplot as plt
from skimage import img_as_ubyte


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def printRandom3DSample(Arr, numElts=(5, 2, 3)):
    # A is a 3d-array
    # Total number of elements: numElts[0]*numElts[1]*numElts[2]
    # constraints:
    #    - numElts[0] < Arr.shape[0]
    #A = np.random.randint(5, size=(2, 4, 5)) 
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
    
    # hb[0] = 0; hg[0] = 0; hr[0] = 0
    # plot_histogram(hb, title="hb", block=False)
    # plot_histogram(hg, title="hg", block=False)
    # plot_histogram(hr, title="hr", block=True)
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


def resizeImg(src, factor=0.3):
    #percent by which the image is resized
    if int(factor) == 1:
        return src

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * factor)
    height = int(src.shape[0] * factor)

    # dsize
    dsize = (width, height)
    # resize image
    out = cv.resize(src, dsize)
    return out

def convertDouble2Uint8(img):
    
    a = max(abs(img.max()), abs(img.min()))
    print(img.min(), img.max(), "a: ", a)
    # _min = 0
    # _max = a
    # minMax = 255.0 / (_max - _min)
    # img = img - _min
    # img = img * minMax
    # cv_image = np.uint8(img)

    _min = -a
    _max = a
    minMax = (_max - _min)
    img = 2*(img - _min)
    img = (img / minMax) -1
    cv_image = img_as_ubyte(img)
    return cv_image

def computeGrad(img):
    # Compute gradient of the two images.
    ##** Manual gradient 'Forward'
    # out_x[:,0:-1,:] = img[:,1:,:].astype('double') - img[:,0:-1,:].astype('double')
    # out_y[0:-1,:,:] = img[1:,:,:].astype('double') - img[0:-1,:,:].astype('double')
    
    ##** Manual gradient 'Centered'
    out_x = np.zeros(img.shape, np.double)
    out_y = np.zeros(img.shape, np.double)
    out_x[:,1:-1,:] = 0.5*(img[:,2:,:].astype('double') - img[:,0:-2,:].astype('double'))
    out_y[1:-1,:,:] = 0.5*(img[2:,:,:].astype('double') - img[0:-2,:,:].astype('double'))
    
    return out_x, out_y

# define a shortcut for the Fourier tran.
def mft(U):
    return np.fft.fftshift(np.fft.fft2(U))

# define a shortcut for the inverse Fourier tran.
def imft(U):
    return np.fft.ifft2(np.fft.ifftshift(U)).real

def poissonSolver(gx, gy):
    print("poissonSolver")
    # Initialization of the output image
    I = np.zeros(gx.shape) # zeros(size(gx)); % init.
    
    # Extend the gradient G (g'(-x) = g'(x)). rev(x)=reverse(x)    
#    gx = np.hstack((gx,gx[:,::-1,:])) # gx = [gx  rev(gx)]    
#    gx = np.vstack((gx,gx[::-1,:,:])) # gx = [gx; rev(gx)]
    # cv.imshow("gx-1", resizeImg(gx))    
    
#    gy = np.vstack((gy,gy[::-1,:,:])) # gy = [gy; rev(gy)]    
#    gy = np.hstack((gy,gy[:,::-1,:])) # gy = [gy rev(gy)];
    # cv.imshow("gy-1", resizeImg(gy))
    
    H,W,C = gx.shape
    # I2 = np.zeros(gx.shape)
    
    ##  Define frequency domain,
    wx, wy = np.meshgrid(np.arange(1, W+1, 1), np.arange(1, H+1, 1))
    
    wx0 = np.floor(W/2)+1
    wy0 = np.floor(H/2)+1 # zero frec
    wx = wx - wx0
    wy = wy - wy0
    cv.imshow("Grad gx", resizeImg(gx, 0.5))#resizeImg(I, 0.4))

    cx = ((1j*2*np.pi)/W)*wx
    cy = ((1j*2*np.pi)/H)*wy
    d = (cx)**2 + (cy)**2;print("---", gx.shape)
    
    print("**********Print zeros : ", np.argwhere(np.abs(d) == 0), ", \
    Center: ", int(wy0), int(wx0))
    
    del wx, wy
    
    for c in range(0, C):
        Gx = gx[:,:,c]
        Gy = gy[:,:,c]
        
        Vx = (cx) * mft(Gx)
        Vy = (cy) * mft(Gy)

        # FT_I = ( Vx + Vy ) / ( d )
        FT_I = np.zeros_like(Vx)
        np.divide(( Vx + Vy ), d, out=FT_I, where=d!=0) #only divide nonzeros else 1

        FT_I[int(wy0-1), int(wx0-1)] = 10 # set DC value (undefined in the previous div.)

        Aux    = imft(FT_I)
#        I[:,:,c]  = Aux[0:int(H/2), 0:int(W/2)] # keep the original portion of the space.
        I[:,:,c]  = Aux[0:H, 0:W] # keep the original portion of the space
        # I2[:,:,c]  = Aux
        del Gx, Gy, FT_I, Aux, Vx, Vy

    # I2 = resizeImg(I2, 0.4)
    # cv.imshow("I2", I2/255.0)
    
    cv.normalize(I, I, 0, 1, cv.NORM_MINMAX)
    return I#/255.0

def make_data_3D(x_range, y_range, step):
     X = np.arange(-x_range, x_range, step)
     Y = np.arange(-y_range, y_range, step)
     X, Y = np.meshgrid(X, Y)
     R = np.sqrt(X**2 + Y**2)
     Z = np.sin(R)
     return X, Y, Z

if __name__=="__main__":

    # setting the id of the dataset to process
    idx = 0
    m_param = [
        {
        'label': 'Labs',
        # 'dir':"/media/sf_Data/poisson_editing_stitching/images/tk_img",
        'dir':"data/Labs",
        'num_frames': 4,
        'scale_factor': 0.5
        },
        {
        'label': 'Airplanes',
        # 'dir':"/media/sf_Data/data_stitching/Airplanes/m_result/1_res_param_all_C04_D95_S01/tk_img",
        'dir':"data/Airplanes",
        'num_frames': 5,
        'scale_factor': 0.2
        },
    ]
    label = m_param[idx]['label']
    folder = m_param[idx]['dir']
    num_frames = m_param[idx]['num_frames']
    scf = m_param[idx]['scale_factor']
    
    TkImgs = []
    # RkImgs = []
    PkImgs = []
    # Grads = []
    Grads_crop = []
    
    for i in range(num_frames):
        tkImg = cv.imread('{folder}/tk_{idx}.png'.format(folder=folder, idx=i) )
        # rkImg = cv.imread('{folder}/rtk_{idx}.png'.format(folder=folder, idx=i) )
        pkImg = cv.imread('{folder}/ptk_{idx}.png'.format(folder=folder, idx=i) )
        
        ## Pre-processing: Replace white into black
        tkImg[tkImg == 255] = 0;
        # rkImg[rkImg == 255] = 0;
        pkImg[pkImg == 255] = 0;
        
        # Resize input images
        tkImg = resizeImg(tkImg, scf)
        # rkImg = resizeImg(rkImg, scf)
        pkImg = resizeImg(pkImg, scf)
        
        grad_x, grad_y = computeGrad(tkImg)
        # Grads.append((grad_x, grad_y))
        
        TkImgs.append(tkImg)
        # RkImgs.append(rkImg)
        PkImgs.append(pkImg)
        
        # Crop gradient
        Omegas  = ~(PkImgs[i] == 0);
        grad_x_c = grad_x * Omegas
        grad_y_c = grad_y * Omegas
        Grads_crop.append((grad_x_c, grad_y_c))
        print("Iteration: {} done...".format(i))
        # Clear memory
        del Omegas, grad_x_c, grad_y_c, grad_x, grad_y, tkImg, pkImg
    
    # print('Tk.shape: {0}, Rk.shape: {1}, Pk.shape: {2}'.format(len(TkImgs), len(RkImgs), len(PkImgs)) )
    print('Tk.shape: {0}, Pk.shape: {1}'.format(len(TkImgs), len(PkImgs)) )
    print("Images Loaded ...")
    
    T = np.zeros_like(PkImgs[0])
    for i in range(num_frames):
        T = T + PkImgs[i];
    
    # hb, hg, hr = generate_combined_histogram(PkImgs)
    
    del PkImgs, TkImgs
    cv.imshow('Sum_img', T)
    print(T.shape, "-", "Plotting simple merge done ...")
    
    # gr_hist, gr_img = generate_histogram(T, do_print=True)
    
    ## Combine gradient
    gradx, grady = Grads_crop[0][0], Grads_crop[0][1]
    for i in range(1, num_frames):
        # # %1- Select where |G| < |Gradients{i}|
        # # %2. Replace the value of |G| where we have max gradient
        gradx = np.where(np.abs(gradx) >= np.abs(Grads_crop[i][0]), gradx, Grads_crop[i][0])
        grady = np.where(np.abs(grady) >= np.abs(Grads_crop[i][1]), grady, Grads_crop[i][1])
    
    del Grads_crop
    
    cv.imshow('Combine Gradients_X', resizeImg(gradx, 0.4))
    cv.imshow('Combine Gradients_Y', resizeImg(grady, 0.4))

    # printRandom3DSample(gradx,numElts=(8, 20, 3))
    # printRandom3DSample(grady,numElts=(8, 20, 3))

    # print("Sleeping for 10 sec before starting the blending process")
    # time.sleep(10)

    I = poissonSolver(gradx, grady)
    print("Poisson Blending Done...")
    cv.imshow("Final Img", I)#resizeImg(I, 0.4))
    
    filename = '{folder}/blending_results_{lbl}.png'.format(folder=folder, lbl=label)
    filename_hist = '{folder}/blending_results_hist_{lbl}.png'.format(folder=folder, lbl=label)
    img = convertDouble2Uint8(I)
    # img_hist = apply_histogram(img, histo=(hb, hg, hr))#, L = 50
    cv.imwrite(filename, img)
    # cv.imwrite(filename_hist, img_hist)
    cv.namedWindow("Final Img Int", cv.WINDOW_NORMAL)
    # cv.namedWindow("Final Img Hist", cv.WINDOW_NORMAL)
    cv.imshow("Final Img Int", img)
    # cv.imshow("Final Img Hist", img_hist)
    
    cv.waitKey(0)
    cv.destroyAllWindows() #cv.startWindowThread()

















