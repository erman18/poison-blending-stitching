from typing import List

import cv2 as cv
import numpy as np

from img_utils import resizeImg
from poisson import convertDouble2Uint8, poissonSolver
from utils import readFiles, divAB

def computeGrad(img):
    # Compute gradient of the two images.
    out_x = np.zeros(img.shape, np.double)
    out_y = np.zeros(img.shape, np.double)
    out_x[:,1:-1,:] = 0.5*(img[:,2:,:].astype('double') - img[:,0:-2,:].astype('double'))
    out_y[1:-1,:,:] = 0.5*(img[2:,:,:].astype('double') - img[0:-2,:,:].astype('double'))
    
    return out_x, out_y

def computeGuidanceField(grads, masks):
    # Compute the guidance field.
    guide_x = grads[0][0] * masks[0]
    guide_y = grads[0][1] * masks[0]
    for i in range(1, len(grads)):
        gradc_x = grads[i][0] * masks[i]
        gradc_y = grads[i][1] * masks[i]
        guide_x = np.where(np.abs(guide_x) >= np.abs(gradc_x), guide_x, gradc_x)
        guide_y = np.where(np.abs(guide_y) >= np.abs(gradc_y), guide_y, gradc_y)

        # gradc_x = grads[i][0]
        # gradc_y = grads[i][1]
        # guide_x = np.where(np.abs(guide_x) <= np.abs(gradc_x), gradc_x, guide_x) * masks[i] + guide_x * (1 - masks[i])
        # guide_y = np.where(np.abs(guide_y) <= np.abs(gradc_y), gradc_y, guide_y) * masks[i] + guide_y * (1 - masks[i])

    print("Guide X: ", guide_x.shape)
    return guide_x, guide_y

def computeGuidanceFieldAvg(grads, masks):
    # Compute the guidance field.
    guide_x = grads[0][0] * masks[0]
    guide_y = grads[0][1] * masks[0]
    mask_sum = masks[0]
    for i in range(1, len(grads)):
        guide_x += grads[i][0] * masks[i]
        guide_y += grads[i][1] * masks[i]
        mask_sum += masks[i]

    print("Guide X: ", guide_x.shape)
    return divAB(guide_x, mask_sum, fill=0), divAB(guide_y, mask_sum, fill=0)


def main_merge(img_paths: List[str], mask_paths: List[str], out_path, scf=0.5) -> None:    
    # Read images and masks.
    imgs = [resizeImg(cv.imread(img_path), scf) for img_path in img_paths]  
    masks = [resizeImg(cv.imread(mask_path)//255, scf) for mask_path in mask_paths]
    # masks = [~(resizeImg(cv.imread(mask_path)//255, scf) == 0) for mask_path in mask_paths]

    # Compute the sum of the masks.
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((15, 15), np.uint8)
    T = np.zeros_like(imgs[0])
    d_masks = []
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        mask = mask # cv.dilate(mask, kernel, iterations=1)
        d_masks.append(mask)
        T += img * mask
        # cv.imshow(f"Mask + Img : {i}", img * mask)
        # cv.waitKey(0)
    cv.imshow("T", T)
    cv.waitKey(0)

    # Compute gradient of the images.
    grads = [(computeGrad(img)) for img in imgs]

    # Compute the guidance field.
    guide_x, guide_y = computeGuidanceField(grads, d_masks)
    # guide_x, guide_y = computeGuidanceFieldAvg(grads, d_masks)
    cv.imshow("Guide X", resizeImg(guide_x, 0.7))
    cv.imshow("Guide Y", resizeImg(guide_y, 0.7))
    cv.waitKey(0)

    # Compute the Poisson equation.
    img_out = poissonSolver(guide_x, guide_y)
    
    # Convert the double image to uint8.
    img_out = convertDouble2Uint8(img_out)
    
    # Save the result.
    cv.imwrite(out_path, img_out)
    cv.imshow("Result", img_out)
    cv.waitKey(0)


if __name__=="__main__":

    # m_path = "/home/UFAD/enghonda/projects/poison-blending-stitching/data/Labs2"
    m_path = "/home/UFAD/enghonda/projects/poison-blending-stitching/data/Airplanes2"
    # m_path = "/home/UFAD/enghonda/projects/poison-blending-stitching/data/NSH"
    img_paths = readFiles(m_path, "tks")
    mask_paths = readFiles(m_path, "mask")

    main_merge(img_paths, mask_paths, "out.png")
    exit(0)

    # Create the output folder.
    # os.makedirs(m_path, exist_ok=True)

    # setting the id of the dataset to process
    idx = 0
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
    
    TkImgs = [] # list of warped images
    # RkImgs = [] # list of reference images
    PkImgs = [] # list of projected images masked by the seam mask
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
        Omegas  = ~(PkImgs[i] == 0)
        grad_x_c = grad_x * Omegas
        grad_y_c = grad_y * Omegas
        Grads_crop.append((grad_x_c, grad_y_c))
        print("Iteration: {} done...".format(i))

        # Save Omegas
        cv.imwrite('{folder}/Mask_{idx}.png'.format(folder=m_path, idx=i), Omegas*255)

        # Save TkImg
        cv.imwrite('{folder}/Tks_{idx}.png'.format(folder=m_path, idx=i), TkImgs[i])
        
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

















