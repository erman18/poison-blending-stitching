import cv2 as cv
import numpy as np

from poisson import convertDouble2Uint8, poissonSolver
from img_utils import resizeImg

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

if __name__=="__main__":

    # setting the id of the dataset to process
    idx = 1
    m_param = [
        {
        'label': 'Labs',
        'dir':"data/Labs",
        'num_frames': 4,
        'scale_factor': 0.5
        },
        {
        'label': 'Airplanes',
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
        tkImg[tkImg == 255] = 0
        # rkImg[rkImg == 255] = 0
        pkImg[pkImg == 255] = 0
        
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
        T = T + PkImgs[i]
    
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

















