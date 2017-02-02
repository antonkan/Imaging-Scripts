import numpy as np
import tifffile as tif
from skimage import segmentation, morphology, restoration
import skimage
import sys
import os

def sharpen(fp, sigma, weight):
    img = skimage.img_as_float(fp)
    smoothed = skimage.filters.gaussian_filter(fp, sigma);
    p = (img - weight*smoothed)/(1.0 - weight);
    return np.clip(p,0.0,1.0)

def analyze(file):
    outname = file[:-4]
    print 'importing '+ outname + '...'
    data = tif.imread(file)
    
    #smooth both channels with a bilateral filter
    greenS = restoration.denoise_bilateral(data[0],win_size=5,sigma_range=0.31,sigma_spatial=3)
    redS   = restoration.denoise_bilateral(data[1],win_size=5,sigma_range=0.31,sigma_spatial=3)
    
    #sharpen
    greenSharp = sharpen(greenS,40,0.6)
    redSharp=sharpen(redS,40,0.6)
    #isodata thresholding
    greent=greenSharp>skimage.filters.threshold_isodata(greenSharp)
    redt=redSharp>skimage.filters.threshold_isodata(redSharp)
    
    data[0]=255*greent
    data[1]=255*redt
    
    tif.imsave('bin/'+outname+'_bin.tif',data)
    print 'made thresholded image'
    
    #sum both thresholded channels
    combi =  np.add(1*greent,1*redt)>0
    
    colony_area=np.logical_or(greent,redt).sum()

    boundG = segmentation.find_boundaries(greent,mode='thick')
    boundR = segmentation.find_boundaries(redt,mode='thick')
    boundboth = segmentation.find_boundaries(combi,mode='thick')
    
    boundGc = (1*boundG-1*boundboth)>0
    boundRc = (1*boundR-1*boundboth)>0
    
    data[0] = 255*morphology.skeletonize(boundGc)
    data[1] = 255*morphology.skeletonize(boundRc)
    
    tif.imsave('bin/skl/'+outname+'_skl.tif',data)
    
    lengthG = (data[0]==255).sum()
    lengthR = (data[1]==255).sum()
    
    col_diameter = 2*np.sqrt(colony_area/np.pi)
    
    print 'made skeletonized boundary images'
    return lengthG, lengthR, col_diameter


if __name__ == "__main__":
    if not os.path.exists(os.getcwd()+'/bin'):
        os.mkdir('bin')
    if not os.path.exists(os.getcwd()+'/bin/skl'):
        os.mkdir('bin/skl')
    
    if len(sys.argv)>=2:
        input = sys.argv[1]
        if '.tif' in input:
            g,r,d = analyze(input)
            print 'boundary length = ', g, ' ,', r, ', colony diameter = ', d
    else:
        fout = open('analysisResults.csv','w')
        fout.write(', Green length, Red length, Colony diameter\n')
        for f in os.listdir(os.getcwd()):
            if 'tif' in f:
                g,r,d = analyze(f)
                print 'boundary length = ', g, ' ,', r, ', colony diameter = ', d
                fout.write(f + ', ' + str(g) + ', '+str(r) + ', '+str(d)+'\n')
    fout.close()
    print 'All done :)'

