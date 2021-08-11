from astropy.io import fits
import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
from lmfit import Model, Parameter
import scipy.integrate as integrate
from scipy.stats import median_abs_deviation as mabsdev
import math
from ew_functions import *



def gaussian(x, cont, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return cont + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

#--------------------------------------



def slopedgaussian(x, slope, cont, amp, cen, wid):
    """1-d gaussian: gaussian(x, slope, cont, amp, cen, wid)"""
    return slope*x + cont + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


#--------------------------------------



def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w



#--------------------------------------


def lidopplershift(wavelength, flux, error, amp):
    """
    Corrects flux and wavelength for doppler shift of lithium line by Gaussian fitting the H-alpha line.
    Inputs:
    wavelength         Wavelength extracted from FITS file.
    flux               Flux extracted from FITS file.
    amp                Amplitude estimate to be passed to Gaussian fit.

    Outputs:
    shiftliwv          Lithium wavelength corrected for Doppler shift.
    liflx              Lithium flux corrected for Doppler shift.

    """

    # Crop data to H-alpha range
    ha_lambcrop = np.where((wavelength > 655.08) & (wavelength < 657.48))
    
    lamb = wavelength[ha_lambcrop]
    flx = flux[ha_lambcrop]

    # Divide data by two to get rid of overlap
    halfdata = len(lamb)//2

    wvlen = lamb[:halfdata+1]
    flx = flx[:halfdata+1]
    

    smthflx = moving_average(flx,4)
    smthwvlen = moving_average(wvlen,4)


    # Fit the Gaussian and output
    gmodel = Model(gaussian)
    result = gmodel.fit(smthflx, x=smthwvlen, cont=0.85, amp=amp, cen=656.3, wid=0.1)

    values = []
    for param in result.params.values():
        values.append(param.value)
    
    # center of fit
    cend = values[2]
    cen = 656.28


    wvlenshift = cen - cend
    wvlens = wvlenshift + smthwvlen
    c = 2.99792e5
    ds = -np.log(wvlens/smthwvlen)*c

    # Crop data for Li range
    li_lambcrop = np.where((wavelength > 669) & (wavelength < 672))
    # And make sure it is the same length as the H-alpha data
    liwv = wavelength[li_lambcrop]
    liwv = liwv[:len(ds)]
    liflx = flux[li_lambcrop]
    liflx = liflx[:len(ds)]
    lierr = error[li_lambcrop]
    lierr = lierr[:len(ds)]
    shiftliwv = liwv*np.exp(-ds/c)

    return shiftliwv, liflx, lierr



#--------------------------------------



def gaussianfit(wavelength, flux):
    """
    Function fits a gaussian to the data for the purpose of creating quicker paramater guesses.
    
    Inputs:     
    wavelength         Wavelength extracted from FITS file.
    flux               Flux extracted from FITS file.
    
    Output:
    values             An array containing all the values of the variables used in slopedgaussian.
    """

    gmodel = Model(slopedgaussian)

    # do fit
    result = gmodel.fit(flux, x=wavelength, slope=0, cont=0.6, amp=Parameter("amp", value=-0.3, max=1e-5), cen=Parameter("cen", value=670.78, min=670.7, max=670.85), wid=Parameter("wid", value=0.08, min=1e-2, max=0.09))


    # Take values of fitted Gaussian
    values = []
    for param in result.params.values():
        values.append(param.value)

    return values 



#--------------------------------------




def calculate_ew(wavelength, flux, params):
    """
    Function which takes FITS files and calculates the Li equivalent widths. 
    Saves a figure and returns an array containing file name and equivalent widths.

    Inputs:
    wavelength         Wavelength extracted from FITS file and shifted to correct for Doppler shift.
    flux               Flux extracted from FITS file.
    params             An array containing all the initial guesses of the variables used in slopedgaussian.       

    Output:
    ewma               Equivalent width in milli-Angstroms
    
    
    """

    gmodel = Model(slopedgaussian)
    
    slope = params[0]
    cont = params[1]
    amp = params[2]
    wid = params[4]
    cen = params[3]

    # do fit
    result = gmodel.fit(flux, x=wavelength, slope=slope, cont=cont, amp=Parameter("amp", value=amp, max=1e-5), cen=Parameter("cen", value=cen, min=670.7, max=670.85), wid=Parameter("wid", value=wid, min=1e-2, max=0.09))

    
    # Take values of fitted Gaussian
    values = []
    for param in result.params.values():
        values.append(param.value)

    slope = values[0]
    cont = values[1]
    amp = values[2]
    cen = values[3]
    wid = values[4]


    # 2 curves to calculate area between (area of the Gaussian)
    gauss = lambda x: slope*x + cont + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

    continuum = lambda x: slope*x + cont

    x = np.linspace(670, 671.5, 500)

    fx = gauss(x)
    gx = continuum(x)

    fdu = integrate.quad(gauss, 670, 671.5)[0]
    gdu = integrate.quad(continuum, 670, 671.5)[0]

    area = gdu - fdu

    rng = np.where((gx-fx)> 9e-3)[0]
    if len(rng) == 0:
        ewma = float("nan")
    else:

        start = rng[0]
        end = rng[-1]

        # Solving for equivalent width using area of trapezoid
        a = gx[start]
        b = gx[end]

        ew = 2*area/(a+b)
        ewma = ew*10000 

    return ewma



#-------------------------------------



def ewerror(wavelength, flux, error, step):
    """
    This function calculates the error in the Lithium EW.
    
    Inputs:
    wavelength          Wavelength extracted from FITS file and shifted to correct for Doppler shift.
    flux                Flux extracted from FITS file.
    error               Error of the flux.
    step                Number of Monte Carlo loops to run.
    
    Output:
    ewstdev             Standard deviation of the EWs calculated using Monte Carlo.
    ewmabsdev           Median absolute deviation of the EWs calculated using Monte Carlo.
    nannum              Number of nans encountered when calculating EWs.
    
    """
    
    
    # run Gaussian fit on base data
    values = gaussianfit(wavelength, flux)

    # create noise for Monte Carlo
    noise = np.random.normal(0, 1, (step, len(wavelength)))

    # new noisy data
    sim_flux = noise*error + flux


    noisy_ews = np.array([])
    nannumb = 0
    # iterate over rows and calculate EWs
    for row in sim_flux:
        noisy_ew = calculate_ew(wavelength, row,  values)
        if math.isnan(noisy_ew):
            nannumb+=1
            
        noisy_ews = np.append(noisy_ews, noisy_ew)

    #noisy_ews_noout = reject_outliers(noisy_ews)
    ewstdev = np.nanstd(noisy_ews)
    ewmabsdev = mabsdev(noisy_ews, scale="normal", nan_policy = "omit")
   
    return ewstdev, ewmabsdev, nannumb



#--------------------------------------



