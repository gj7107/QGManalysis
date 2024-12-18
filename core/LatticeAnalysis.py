import scipy.io, scipy
import numpy as np
import random
from skimage import restoration
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
import time
import core.richardson_lucy as rl
import torch

# Atom occupancy analysis   
class LatticeAnalysis:
    def __init__(self, analysisparameters) :
        self.AnalysisParameters = analysisparameters
        self.__nLat = 150
        self.__lensamples = 10000
        self.__lowerbound = -20
        
    #### PRIVATE FUNCTIONS ####
    def __readfile(self, filepath) :
        # File read
        print("File read ...",filepath)
        params = self.AnalysisParameters
        f = open(filepath, "rb")
        arr = np.fromfile(f, dtype=np.uint32)
        imgsize = params['imgsize']
        f.close()
        
        try :
            arr = np.reshape(arr, [params['n']+1,imgsize[0], imgsize[1]])
        except:
            return np.zeros([params['n']+1,imgsize[0], imgsize[1]])
        
        return arr

    def __cropimg(self, arr) :
        n = self.AnalysisParameters['n']
        ROI = self.AnalysisParameters['ROI']
        retarr = np.zeros((n,ROI[0]*2,ROI[0]*2),dtype=float)
        rangex = np.arange(ROI[1]-ROI[0],ROI[1]+ROI[0])
        rangey = np.arange(ROI[2]-ROI[0],ROI[2]+ROI[0])
        for i in range(n) :
            Arri = np.squeeze(arr[i,:,:]).astype(float)
            Arrn = np.squeeze(arr[n,:,:]).astype(float)
            retarr[i,:,:] = Arri[rangey,:][:,rangex] - Arrn[rangey,:][:,rangex] - 0.0
        self.__minvalue = retarr.min()
        self.__maxvalue = retarr.max()
        retarr[retarr < self.__lowerbound] = 0
        self.__minvalue = self.__lowerbound

        return retarr
    
    def __errfunction(self, x, Mat, centers, a1, a2, x_lmin, y_lmin) :
        dx, dy = x
        #a1 = np.array([[norm1*np.cos(th1)],[norm1*np.sin(th1)]])
        #a2 = np.array([[norm2*np.cos(th2)],[norm2*np.sin(th2)]])
        #direction1 = a1 / np.linalg.norm(a1)
        #direction2 = a2 / np.linalg.norm(a2)
        #M = np.array([[1, -np.dot(direction1, direction2)],[-np.dot(direction1, direction2), 1]]) / (1 - np.dot(direction1, direction2)**2)
        #V = np.array([[direction1[0],direction1[1]],[direction2[0],direction2[1]]])
        pos = Mat @ centers

        # Compute alphas more efficiently
        alphas_x = pos[0, :] / np.linalg.norm(a1) - x_lmin - dx
        alphas_y = pos[1, :] / np.linalg.norm(a2) - y_lmin - dy
        alphas = np.stack((alphas_x, alphas_y), axis=0)

        # Compute dalpha in a vectorized manner
        dalpha = alphas - np.round(alphas)

        # Calculate ret in a single step without reshaping
        ret = np.mean(dalpha**2)

        return ret

    def __saveData(self, outputfilename, Lats, xopts, photoncounts) :
        lattice_a = self.AnalysisParameters['lattice_a']
        a1 = lattice_a[:,0]
        a2 = lattice_a[:,1]
        th1 = np.arctan2(a1[1],a1[0])
        th2 = np.arctan2(a2[1],a2[0])
        norm1 = np.linalg.norm(a1)
        norm2 = np.linalg.norm(a2)
        
        n = self.AnalysisParameters['n']
        params = np.zeros((n,6))
        
        for i in range(n) :
            params[i,0] = th1
            params[i,1] = th2
            params[i,2] = norm1
            params[i,3] = norm2
            params[i,4] = xopts[i][0]
            params[i,5] = xopts[i][1]
        
        Lats_i = np.zeros((1,2),dtype=object)
        Lats_i[0,0] = np.array(Lats[0])
        Lats_i[0,1] = np.array(Lats[1])
        
        savemat(outputfilename,{'Lats_i':Lats_i,'params':params,'photoncounts':photoncounts})    

        return 0
    
    # Find Phase of the optical lattice
    def FindPhase(self, imgc) :
        lattice_a = self.AnalysisParameters['lattice_a']
        a1 = lattice_a[:,0]
        a2 = lattice_a[:,1]
        th1 = np.arctan2(a1[1],a1[0])
        th2 = np.arctan2(a2[1],a2[0])
        norm1 = np.linalg.norm(a1)
        norm2 = np.linalg.norm(a2)
        direction1 = a1 / norm1
        direction2 = a2 / norm2
        M = np.array([[1, -np.dot(direction1, direction2)],[-np.dot(direction1, direction2), 1]]) / (1 - np.dot(direction1, direction2)**2)
        V = np.array([[direction1[0],direction1[1]],[direction2[0],direction2[1]]])
        T = np.array([[norm1, 0],[0,norm2]])
        
        
        #x0 = [dx, dy]
        _, wimg, himg = imgc.shape
        p1 = np.array([[0],[0]])
        p2 = np.array([[wimg-1],[0]])
        p3 = np.array([[0],[himg-1]])
        p4 = np.array([[wimg-1],[himg-1]])
        
        p1 = M @ V @ p1 #np.array([[np.dot(p1, direction1)],[np.dot(p1,direction2)]])
        p2 = M @ V @ p2
        p3 = M @ V @ p3
        p4 = M @ V @ p4
        #p2 = M @ np.array([[np.dot(p2, direction1)],[np.dot(p2,direction2)]])
        #p3 = M @ np.array([[np.dot(p3, direction1)],[np.dot(p3,direction2)]])
        #p4 = M @ np.array([[np.dot(p4, direction1)],[np.dot(p4,direction2)]])
        
        x_l1 = p1[0] / norm1
        x_l2 = p2[0] / norm1 
        x_l3 = p3[0] / norm1
        x_l4 = p4[0] / norm1
        y_l1 = p1[1] / norm2
        y_l2 = p2[1] / norm2
        y_l3 = p3[1] / norm2 
        y_l4 = p4[1] / norm2
        
        x_lmin = np.floor(np.min([x_l1, x_l2, x_l3, x_l4]))
        y_lmin = np.floor(np.min([y_l1, y_l2, y_l3, y_l4]))
        x_lmax = np.ceil(np.max([x_l1, x_l2, x_l3, x_l4]))
        y_lmax = np.ceil(np.max([y_l1, y_l2, y_l3, y_l4]))
        
        print(x_lmin, y_lmin, x_lmax, y_lmax)
                
        [X,Y] = np.meshgrid(range(wimg), range(himg))
        
        n = self.AnalysisParameters['n']
        PSF = self.AnalysisParameters['PSF']        
        imgth = self.AnalysisParameters['imgth']
        
        Js = []
        xopts = []
        for i in range(n) :
            img = np.squeeze(imgc[i,:,:]) - self.__minvalue
            imgnorm = img/img.max()
            device = "cuda"
            t0 = time.time()
            ref = torch.from_numpy(np.array(imgnorm)).unsqueeze(0).unsqueeze(0).to(device).float()
            x_0 = torch.ones_like(ref)
#            k_size = PSF.shape[0]
            k_ref = torch.from_numpy(PSF).unsqueeze(0).unsqueeze(0).float().to(device)
            res = rl.richardson_lucy(ref, x_0, k_ref, steps=100, tv=False) 
            J = res[0, 0].detach().cpu().numpy()
            
            print("Richardson-Lucy, ", time.time() - t0)
            ys = X[J > imgth]
            xs = Y[J > imgth]
            
            lensamples = self.__lensamples
            if(len(ys) < lensamples) :
                lensamples = len(ys)
            sampleidx = random.sample(range(len(ys)),lensamples)
            ys = ys[sampleidx]
            xs = xs[sampleidx]
            centers = np.vstack((xs,ys))
            if(i == 0) :
                initialvalue = [0, 0]
                ldf = lambda x: self.__errfunction(x,M@V,centers,a1,a2,x_lmin,y_lmin)
                xopt = scipy.optimize.fmin(func=ldf, x0=initialvalue)
            else :
                px = self.AnalysisParameters['px']
                py = self.AnalysisParameters['py']
                funpx = np.poly1d(px)
                funpy = np.poly1d(py)
                xopt = [funpx(i+1), funpy(i+1)]
            Js.append(J)
            xopts.append(xopt)
        return Js, M, V, T, xopts, x_lmin, y_lmin
    
    '''
    # gather data into the lattice sites
    def LatticeReconstruction(self, J, M, V, T, xopt, x_lmin, y_lmin) :
        dx = xopt[0]
        dy = xopt[1]
        [x,y] = np.meshgrid(range( self.__nLat +1),np.arange(-np.floor( self.__nLat /2),np.ceil( self.__nLat /2),1))
        Lat = np.zeros(x.shape)
        for i in range(x.shape[0]) :
            for j in range(y.shape[0]) :
                t = np.linalg.solve(M@V,T@(np.array([[i+dx+x_lmin],[j+dy+y_lmin]])))
                r1 = int(np.floor(t[0,:]-1))
                c1 = int(np.floor(t[1,:]-1))
                r2 = int(np.ceil(t[0,:]+1))
                c2 = int(np.ceil(t[1,:]+1))
                if(r1 < 0 or r2 >= J.shape[0]) :
                    continue
                if(c1 < 0 or c2 >= J.shape[1]) :
                    continue
                Lat[i,j] = np.mean(J[r1:r2,c1:c2])
                
        return Lat
    '''
    def LatticeReconstruction(self, J, M, V, T, xopt, x_lmin, y_lmin):
        dx = xopt[0]
        dy = xopt[1]
        x, y = np.meshgrid(range(self.__nLat + 1), np.arange(-np.floor(self.__nLat / 2), np.ceil(self.__nLat / 2), 1))
        Lat = np.zeros(x.shape)
        
        # Precompute the transformed coordinates
        coords = np.vstack((x.flatten() + dx + x_lmin, (y.flatten() + self.__nLat/2) + dy + y_lmin))
        t_coords = np.linalg.solve(M @ V, T @ coords)

        r1 = np.floor(t_coords[0, :] - 1).astype(int)
        c1 = np.floor(t_coords[1, :] - 1).astype(int)
        r2 = np.ceil(t_coords[0, :] + 1).astype(int)
        c2 = np.ceil(t_coords[1, :] + 1).astype(int)
        
        # Ensure the indices are within bounds
        valid_indices = (r1 >= 0) & (r2 < J.shape[0]) & (c1 >= 0) & (c2 < J.shape[1])
        
        for idx in np.where(valid_indices)[0]:
            Lat.flat[idx] = np.mean(J[r1[idx]:r2[idx], c1[idx]:c2[idx]])
        
        return Lat

    # Compensate inhomogenieoty from the Beam intensity
    def IntensityCalibration(self, ndiv, Lat) :
        w, h = Lat.shape
        IntensityMap = np.zeros((w,h))
        for i in range(ndiv) : 
            for j in range(ndiv) :
                idx_is = int(i*w/ndiv)
                idx_ie = int((i+1)*w/ndiv)
                idx_js = int(j*h/ndiv)
                idx_je = int((j+1)*h/ndiv)
                AOI = Lat[idx_is:idx_ie,:][:,idx_js:idx_je]
                IntensityMap[idx_is:idx_ie,:][:,idx_js:idx_je] += AOI.max()
        IntensityMap[IntensityMap < IntensityMap.max() * 0.5] = IntensityMap.max()
        return IntensityMap
    
    def GetPhotonCount(self, img) :
        return np.mean(img[img > self.__maxvalue* 0.4]).item()
    
    # Process lattice
    def ProcessLattice(self, filepath) :
        t0 = time.time()
        print("Read file ...", 'time = ', time.time() - t0) 
        img = self.__readfile(filepath)
        imgc = self.__cropimg(img)
        print("File shape", img.shape, " Coppred to ", imgc.shape, "Min/Max = ", self.__minvalue , "/", self.__maxvalue, 'time = ', time.time() - t0)
        print("Finding lattice phase ...", 'time = ', time.time() - t0)
        Js, M, V, T, xopts, x_lmin, y_lmin = self.FindPhase(imgc)
        print("Optimized output: ", xopts[0][0], xopts[0][1], 'time = ', time.time() - t0)
        n = self.AnalysisParameters['n']
        Lats = []
        for i in range(n) : 
            Lat = self.LatticeReconstruction(Js[i], M, V, T, xopts[i], x_lmin, y_lmin)
            Lats.append(Lat)
            
        Imap = self.IntensityCalibration(self.AnalysisParameters['ndiv'], Lats[0])
        
        if self.AnalysisParameters['BeamIntensityCalibration'] : 
            for i in range(n) :
                Lats[i] = Lats[i] / Imap
        photoncounts = [self.GetPhotonCount(imgc[0,:,:]), self.GetPhotonCount(imgc[1,:,:])]
        print("Photon count :", photoncounts, 'time = ', time.time() - t0)
        root, ext = os.path.splitext(filepath)
        self.__saveData(root+'_Lat.mat', Lats, xopts, photoncounts)
        
        return Lats, photoncounts, xopts
        # ROI cropping
        
    

    
def GetTemp(Lat, draw) :
    kB = 1.38e-23
    hbar = 1.05457e-34
    m = 7 * 1.6605e-27
    
    r, c = np.where(Lat> 0.5)
    com = [np.mean(r), np.mean(c)]
    x0 = com[1]
    y0 = com[0]
    
    radius = np.floor(np.min([Lat.shape[0] - y0, y0-1,Lat.shape[1] - x0, x0-1]))
    cropMat = Lat[int(np.round(y0)-radius):int(np.round(y0)+radius),:][:,int(np.round(x0)-radius):int(np.round(x0)+radius)]
    xData = np.arange(cropMat.shape[1])
    radialCut = np.zeros((1,cropMat.shape[1]))
    for th in range(180) :
        rotimg = scipy.ndimage.rotate(cropMat, th, reshape=False, order=1)
        radialCut += rotimg[int(radius),:]/180
    
    radialCut = radialCut.reshape(-1)
    profileX = range(cropMat.shape[0])
    profileY = np.sum(cropMat, axis=1)

    def gaussFit(x, profileX, profileY) :
        a = x[0]
        b = x[1]
        c = x[2]
        y_ = a*(1-(profileX-b)**2/c**2) * (profileX>(b-c)) * (profileX<(b+c))
        return np.sum((profileY-y_)**2)
    popt = scipy.optimize.fmin(lambda x: gaussFit(x,profileX,profileY),x0=[profileY.max(),radius,radius])
    MottRadius = popt[2]
    print(popt)       
    
    def drawGaussFit(popt) :
        a = popt[0]
        b = popt[1]
        c = popt[2]
        xx = np.linspace(np.min(profileX),np.max(profileX))
        y_ = a*(1-(xx-b)**2/c**2) * (xx>(b-c)) * (xx<(b+c))
        plt.plot(xx, y_)
        return
    
    
    mu = 1.2
    T = 0.13
    Rtf = MottRadius
    x0 = [mu, T, Rtf]
    xmin = [0.2,0.02,10]
    xmax = [1.8,0.8,Rtf+25]
    
    cen_radialcut = int(np.round((len(radialCut)-1)/2))
    radialCut_excluded = np.hstack((radialCut[25:cen_radialcut-2],radialCut[cen_radialcut+2:-25]))
    xData_excluded = np.hstack((xData[25:cen_radialcut-2],xData[cen_radialcut+2:-25]))
    
    def MottAverage(x, mu,T,Rtf) :
        dataSize = len(x)
        Z = np.zeros((dataSize,dataSize)) 
        nMap = np.zeros((dataSize, dataSize))
        x0 = int(np.round((dataSize-1)/2))
        y0 = int(np.round((dataSize-1)/2))
        gridX, gridY = np.meshgrid(range(dataSize), range(dataSize))
        potential = mu * ((gridX - x0)**2 + (gridY - y0)**2) / Rtf**2
        for n in range(12) :
            partition = np.exp((mu-potential)*n/T - n * (n-1) /2 / T)
            Z += partition
            nMap += np.mod(n, 2) * np.exp((mu-potential)*n/T - n * (n-1) /2 / T)
            
        nMap = nMap / Z
        radialCut = np.zeros((1,nMap.shape[1]))
        for th in range(180) :
            rotimg = scipy.ndimage.rotate(nMap, th, reshape=False, order=1)
            radialCut += rotimg[y0,:]/180            
        radialCut = radialCut.reshape(-1)
        cen_radialcut = int(np.round((len(radialCut)-1)/2))
        radialCut_excluded = np.hstack((radialCut[25:cen_radialcut-2],radialCut[cen_radialcut+2:-25]))
        return radialCut_excluded   
    
    xopt, pcov = scipy.optimize.curve_fit(MottAverage,xData,radialCut_excluded, p0=x0,bounds=(xmin,xmax))
    yFit = MottAverage(xData, xopt[0], xopt[1], xopt[2])
    
    if draw : 
        plt.figure()
        plt.subplot2grid((2,2),(1,1))
        drawGaussFit(popt)
        plt.plot(profileX,profileY)
        
        plt.subplot2grid((2,2),(0,0),colspan=2)
        plt.plot(xData_excluded, radialCut_excluded, 'bo')
        plt.plot(xData_excluded, yFit, 'r-')
        
        plt.subplot2grid((2,2),(1,0))
        plt.imshow(cropMat)
        
    return xopt