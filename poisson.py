import cv2 as cv
import numpy as np
from skimage import img_as_ubyte


def convertDouble2Uint8(img):
    
    a = max(abs(img.max()), abs(img.min()))
    print(img.min(), img.max(), "a: ", a)

    _min, _max = -a, a
    minMax = (_max - _min)
    img = 2*(img - _min)
    img = (img / minMax) -1
    cv_image = img_as_ubyte(img)
    return cv_image

# define a shortcut for the Fourier tran.
def mft(U):
    return np.fft.fftshift(np.fft.fft2(U))

# define a shortcut for the inverse Fourier tran.
def imft(U):
    return np.fft.ifft2(np.fft.ifftshift(U)).real

# Expand the image to create symetric to avoid boundary problems
def expandImage(original_image, d="x"):
    # Get the dimensions of the original image
    rows, cols, ch = np.shape(original_image)

    # Create an empty array to hold the final image
    final_image = np.empty((rows * 2, cols * 2, ch), dtype=original_image.dtype)

    if d == "x":
        # Place the original and mirror images in the final image
        final_image[0:rows, 0:cols, :] = original_image
        final_image[0:rows, cols:cols*2, :] = -np.flip(original_image, axis=1)
        final_image[rows:rows*2, 0:cols, :] = np.flip(original_image, axis=0)
        final_image[rows:rows*2, cols:cols*2, :] = -np.flip(original_image, axis=(0,1))
    elif d == "y":
        # Place the original and mirror images in the final image
        final_image[0:rows, 0:cols, :] = original_image
        final_image[0:rows, cols:cols*2, :] = np.flip(original_image, axis=1)
        final_image[rows:rows*2, 0:cols, :] = -np.flip(original_image, axis=0)
        final_image[rows:rows*2, cols:cols*2, :] = -np.flip(original_image, axis=(0,1))

    return final_image

def poissonSolver(gx, gy):
    print("poissonSolver")
    # Initialization of the output image
    I = np.zeros(gx.shape)
    
    # Expand the image to create symetric to avoid boundary problems
    # My solution: function (g'(-x) = g'(x))
    # gx = [gx  gx(:,end:-1:1,:)]; gx = [gx; gx(end:-1:1,:,:)];
    # gy = [gy; gy(end:-1:1,:,:)]; gy = [gy gy(:,end:-1:1,:)];
    gx = expandImage(gx, "x")
    gy = expandImage(gy, "y")

    H,W,C = gx.shape
    
    #  Define frequency domain,
    wx, wy = np.meshgrid(np.arange(1, W+1, 1), np.arange(1, H+1, 1))
    
    wx0 = np.floor(W/2)+1
    wy0 = np.floor(H/2)+1 # zero frec
    wx = wx - wx0
    wy = wy - wy0
    # cv.imshow("Grad gx", resizeImg(gx, 0.5))

    cx = ((1j*2*np.pi)/W)*wx
    cy = ((1j*2*np.pi)/H)*wy
    d = (cx)**2 + (cy)**2;print("---", gx.shape)
    
    print(f"{'-->':>4} Print zeros : {np.argwhere(np.abs(d) == 0)},  Center: ({int(wy0)}, {int(wx0)})")
    
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

        Aux = imft(FT_I)
        I[:,:,c]  = Aux[0:int(H/2), 0:int(W/2)] # keep the original portion of the space.
        # I[:,:,c]  = Aux[0:H, 0:W] # keep the original portion of the space
        # I2[:,:,c]  = Aux
        # del Gx, Gy, FT_I, Aux, Vx, Vy

    # I2 = resizeImg(I2, 0.4)
    # cv.imshow("I2", I2/255.0)
    
    cv.normalize(I, I, 0, 1, cv.NORM_MINMAX)
    return I#/255.0
