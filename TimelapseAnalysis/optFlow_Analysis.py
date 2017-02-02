import numpy as np
import cv2
import tifffile as tif
import lic_internal
import sys
import os
from skimage import filters, morphology


def div2d(field):
    fieldx = field[:,:,0]
    fieldy = field[:,:,1]
    dFx_dx = np.asarray(np.gradient(fieldx))[1,:,:]
    dFy_dy = np.asarray(np.gradient(fieldy))[0,:,:]
    div = dFx_dx + dFy_dy
    return div

def curl2d(field):
    fieldx = field[:,:,0]
    fieldy = field[:,:,1]
    dFy_dx = np.asarray(np.gradient(fieldy))[1,:,:]
    dFx_dy = np.asarray(np.gradient(fieldx))[0,:,:]
    curl = dFy_dx - dFx_dy #in the z-axis
    return curl

def draw_flow(img, flow, step=10):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x-fx/2, y-fy/2, x+fx/2, y+fy/2]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = (np.arctan2(fy, fx) + np.pi)/np.pi
    #bgr = draw_hsv(flow)
    #cv2.polylines(vis, lines, 0, int(bgr))
    for (x1, y1), (x2, y2) in lines:
        color = (0,0,255)
        #color = (int(bgr[x1,y1,0]),int(bgr[x1,y1,1]),int(bgr[x1,y1,2]))
        if np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))>np.sqrt(2):
            cv2.arrowedLine(vis,(x1,y1),(x2,y2),color,1)
        #cv2.circle(vis, (x1, y1), 1, color, -1)
    return vis   #, bgr

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180.0/np.pi/2.0)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*10, 255)
    #print t,np.max(v)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def draw_gradients(flow):
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)
    #print p.shape
    grad = np.gradient(v)
    grad = np.asarray(grad)
    h,w = flow[:,:,0].shape
    bgr = np.zeros((h, w, 3), np.uint8)
    #div = grad[0,:,:,0] + grad[1,:,:,1] #I think this is the diverence
    #print np.min(div), np.max(div)
    #print grad.shape
    bgr[:,:,0] = 0#np.minimum(div*5.0,255)
    bgr[:,:,1] = np.minimum(grad[0,:,:]*5,255)
    bgr[:,:,2] = np.minimum(grad[1,:,:]*5,255)
    res =cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr

def draw_lic(flow,length=31,val=False):# FIXME: This breaks if there are zeros in the flow array
    m,n,two = flow.shape
    fx, fy = flow[:,:,0], flow[:,:,1]
    v = np.sqrt(fx*fx + fy*fy)
    kernellen=length
    kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
    kernel = kernel.astype(np.float32)
    texture = np.random.rand(m,n).astype(np.float32)
    image = lic_internal.line_integral_convolution(flow[:,:,0],flow[:,:,1], texture, kernel)
    #print np.max(image)
    if val:
        return np.asarray(np.clip(0.9*image*v,0,255)).astype(np.uint8)
    else:
        return np.asarray(np.clip(image*4,0,255)).astype(np.uint8)

def convert(img, max=30000.0, min=1000.0):#preprocessing is done here
    denoised = filters.rank.median(img, morphology.disk(1))
    norml = np.clip(1.0*denoised-min,0,max-min)/(max-min) * 255.0
    return np.asarray(norml).astype(np.uint8)

def analyze(f, save=True):
    outname = f[:-4]
    print 'Importing '+outname+'...'
    data = tif.imread(f)# t*512*512 16-bit tiff files
    print 'data input is: ', data.shape, data.dtype
    max = 30000 #default rescaling for histogram
    if '4s-' in outname: max = 46000
    if '4s+' in outname: max = 37000
    if '4ag-' in outname: max = 40000
    if '4ag+' in outname: max = 27000
    if 'G5' in outname: max = 28000
    if 'KAY' in outname: max = 10000
    
    timepoints,m,n = data.shape
    #print 'max is: ', max
    prev = convert(data[0],max) #need to be careful with the type conversion here
    cur_glitch = prev
    print 'data being processed is: ', prev.shape, prev.dtype
    
    flow_vid = np.zeros((timepoints-1,m,n,3)).astype(np.uint8)#RGB image
    warp_vid = np.zeros((timepoints-1,m,n)).astype(np.uint8)#8 bit greyscale
    hsv_vid = np.zeros((timepoints-1,m,n,3)).astype(np.uint8) #RGB image
    lic_vid = np.zeros((timepoints-1,m,n)).astype(np.uint8)#8 bit greyscale
    curl_vid = np.zeros((timepoints-1,m,n,3)).astype(np.uint8)#RGB image red positive, green negative
    div_vid = np.zeros((timepoints-1,m,n,3)).astype(np.uint8)#RGB image red positive, green negative
    
    abs_vort = []
    tot_vort = []
    divs = []
    colony_areas = []
    
    thresh = filters.threshold_otsu(prev)
    for i in range(1,timepoints):
        img = convert(data[i],max)
        flow = cv2.calcOpticalFlowFarneback(prev, img, None, 0.5, 3, 25, 28, 5, 0.7, 256)
        
        inv = 255*np.ones_like(img, np.uint8)
        inv = np.subtract(inv, img)
        cur_glitch = warp_flow(cur_glitch, flow)
        
        #calculate the threshold for pixels containing bacteria - using minimal fluorescence value throughout the timelapse
        if thresh > filters.threshold_otsu(img): thresh = filters.threshold_otsu(img)
        
        cellsnew = img > thresh #binary mask of cells
        cellsold = prev > thresh
        curlarray = np.asarray(curl2d(flow))
        divarray = np.asarray(div2d(flow))
        
        abs_vort.append(np.mean(np.multiply(abs(curlarray),cellsnew)))
        tot_vort.append(np.mean(np.multiply(curlarray,cellsnew)))
        divs.append(np.mean(np.multiply(divarray,cellsnew)))
        colony_areas.append(np.sum(cellsnew))
        colony_radius = np.sqrt(np.sum(cellsnew)/np.pi)
        
        flow_vid[i-1] = draw_flow(inv, flow) #colony image is from current time point, so the flow has just happened
        warp_vid[i-1] = cur_glitch
        hsv_vid[i-1] = draw_hsv(flow)
        lic_vid[i-1] = np.multiply(draw_lic(flow+1e-30,45),cellsnew)
        
        prev = img
        
        grad_mult = 50.0
        
        curl_vid[i-1,:,:,0] = np.multiply(np.multiply(grad_mult*curlarray,cellsnew),curlarray>0).astype(int)
        curl_vid[i-1,:,:,1] = np.multiply(np.multiply(-grad_mult*curlarray,cellsnew),curlarray<0).astype(int)
        div_vid[i-1,:,:,0] = np.multiply(np.multiply(grad_mult*divarray,cellsnew),divarray>0).astype(int)
        div_vid[i-1,:,:,1] = np.multiply(np.multiply(-grad_mult*divarray,cellsnew),divarray<0).astype(int)
        curl_vid[i-1,:,:,2] = img #current time point, so the field is for the velocity that has just happened
        div_vid[i-1,:,:,2] = img #current time point, so the field is for the velocity that has just happened
    
    if save:
        print 'Saving videos... '
        tif.imsave('res/'+outname+'_flow.tif',flow_vid[:]) #flow quiver plot video
        tif.imsave('res/'+outname+'_warp.tif',warp_vid[:]) #warp of initial frame
        tif.imsave('res/'+outname+'_hsv.tif',hsv_vid[:]) #direction mapped to colour, brightness to velocity magnitude
        tif.imsave('res/'+outname+'_lic.tif',lic_vid[:]) #Line integral convolution
        tif.imsave('res/'+outname+'_curl.tif',curl_vid[:])
        tif.imsave('res/'+outname+'_div.tif',div_vid[:])

    return abs_vort, tot_vort, divs, colony_areas

if __name__ == "__main__":
    if not os.path.exists(os.getcwd()+'/res'):
        os.mkdir('res')

    if len(sys.argv)>=2:
        input = sys.argv[1]
        if '.tif' in input:
            analyze(input)
    else:
        fout_curl_abs = open('Data-Vort_abs.csv','w')
        fout_curl = open('Data-curl.csv','w')
        fout_div = open('Data-div.csv','w')
        fout_area = open('Data-ColonyArea.csv','w')

        for f in os.listdir(os.getcwd()):
            if '.tif' in f:
                abs_vort, curl, divs, colony_areas = analyze(f)
                
                fout_curl_abs.write(f[:-4]+', '+str(abs_vort)[1:-1]+'\n')
                fout_curl.write(f[:-4]+', '+str(curl)[1:-1]+'\n')
                fout_div.write(f[:-4]+', '+str(divs)[1:-1]+'\n')
                fout_area.write(f[:-4]+', '+str(colony_areas)[1:-1]+'\n')

        fout_curl_abs.close()
        fout_curl.close()
        fout_div.close()
        fout_area.close()
    print 'All done :)'
