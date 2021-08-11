from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
from matplotlib.ticker import MultipleLocator
import os
from lmfit import Model, Parameter
import scipy.integrate as integrate
from ew_functions import *

#--------------------------------------


def fullspectrumplot(wavelength, flux, error, stardata, title):
    """
    This function creates a plot of the full spectrum provided in the FITS file.
    
    Inputs:
    wavelength     Wavelength in nm
    flux           Outputted flux
    error          Error of the flux
    stardata       Array with filename, star identifier, RA, DEC, spectral type, candidate type, and YSA
    title          Title to name saved figure
    
    Output:
    None
     
    """
    
    plt.clf()
    
    # Figure size
    fig = plt.figure(figsize=(24, 8))
    ax = plt.gca() 
    
    # Plot the data
    plt.plot(wavelength, flux, "k", label="Star Flux")
    plt.plot(wavelength, error, "r", label = "Error")
    plt.legend(loc="upper right")
    
    # Limit on data
    plt.ylim((0, 3))
    plt.xlim((350, 1080))
    
    # Axes labels
    plt.ylabel("Flux", c="k", fontsize=14)
    plt.xlabel("Wavelength (nm)", c="k", fontsize=14)
    
    # Background colour
#     rect = fig.patch  
#     rect.set_facecolor('k')
#     ax.set_facecolor("k")
    
    # Title and subtitle
    subtitle = "RA: " + stardata[2] + "    DEC: " + stardata[3] + "    Spectral Type: " + stardata[5] + "    Candidate Type: " + stardata[6] + "    Young Association: " + stardata[7]
    plt.title(subtitle,fontsize=16, c="k")
    plt.suptitle(stardata[4],fontsize=24, c="k")
    
    
    # Minor axes
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    # Set axes colours
    ax.tick_params(which="major", colors="k", size=10, labelsize=12)
    ax.tick_params(which="minor", colors="k", size=6, labelsize=12)

#     # Box colour
#     ax.spines["bottom"].set_color("white")
#     ax.spines["top"].set_color("white")
#     ax.spines["left"].set_color("white")
#     ax.spines["right"].set_color("white")
    
    # Save plot
    plottitle = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/Full Spectrum Plots/" + title + "fullspectrumplot"
    plt.savefig(plottitle, facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.close()
    
    return



#--------------------------------------


def Lispectrumplot(wavelength, flux, error, stardata, title):
    """
    This function creates a plot of the spectrum provided in the FITS file,
    cropped to the relevant region for Lithium absorption.
    
    Inputs:
    wavelength     Wavelength in nm
    flux           Outputted flux
    error          Error of the flux
    stardata       Array with filename, star identifier, RA, DEC, spectral type, candidate type, and YSA
    title          Title to name saved figure
    
    Output:
    None
      
    """
    
    # Figure size
    plt.clf()
    ax = plt.gca() 
    fig = plt.figure(figsize=(24, 8))
    
    # Plot the data
    plt.plot(wavelength, flux, "k", label="Star Flux")
    plt.plot(wavelength, error, "r", label="Error")
    plt.axvline(x=670.78, color="b", linestyle="dotted", label="Lithium Absorption Wavelength")
    plt.legend(loc="upper right")
    
    # Limits on data
    plt.ylim((0, 1.5))
    plt.xlim((670.6, 671))
    
    # Axes labels
    plt.ylabel("Flux", c="k", fontsize=14)
    plt.xlabel("Wavelength (nm)", c="k", fontsize=14)
    
#     # Background colour
#     rect = fig.patch  
#     rect.set_facecolor('k')
#     ax.set_facecolor("k")
    
    # Title and subtitle
    subtitle = "RA: " + stardata[2] + "    DEC: " + stardata[3] + "    Spectral Type: " + stardata[5] + "    Candidate Type: " + stardata[6] + "    Young Association: " + stardata[7]
    plt.title(subtitle,fontsize=16, c="k")
    plt.suptitle(stardata[4],fontsize=24, c="k")
    
    # Minor axes
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    # Set axes colours
    ax.tick_params(which="major", colors="k", size=10, labelsize=12)
    ax.tick_params(which="minor", colors="k", size=6, labelsize=12)

#     # Box colour
#     ax.spines["bottom"].set_color("white")
#     ax.spines["top"].set_color("white")
#     ax.spines["left"].set_color("white")
#     ax.spines["right"].set_color("white")
    
    # Save plot
    plottitle = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/Li Spectrum Plots/" + title + "Lispectrumplot"
    plt.savefig(plottitle, facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.close()
    
    return


#--------------------------------------


def h_alpha_liplot(wavelength, flux, error, stardata, title):
    """
    This function creates a plot of the spectrum provided in the FITS file,
    cropped to the relevant region for Lithium and H-alpha absorption.
    
    Inputs:
    wavelength     Wavelength in nm
    flux           Outputted flux
    error          Error of the flux
    stardata       Array with filename, star identifier, RA, DEC, spectral type, candidate type, and YSA
    title          Title to name saved figure
    
    Output:
    None
    
    """
    
    # Figure size
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,15))
    
    # Plot the Lithium
    ax1.plot(wavelength, flux, "k", label="Star Flux")
    ax1.plot(wavelength, error, "r", label="Error")
    ax1.axvline(x=670.78, color="purple", linestyle="dotted", label="Lithium Absorption Wavelength")
    ax1.legend(loc="upper right")
    
    # Plot the H-alpha emission
    ax2.plot(wavelength, flux, "k", label="Star Flux")
    ax2.plot(wavelength, error, "r", label="Error")
    ax2.axvline(x=656.28, color="purple", linestyle="dotted", label="H-alpha Absorption Wavelength")
    ax2.legend(loc="upper right")
    
    # Limits on Lithium
    ax1.set_ylim((0, 1.5))
    ax1.set_xlim((670.6, 671))
    
    # Limits on H-alpha
    ax2.set_ylim((0, 1.5))
    ax2.set_xlim((656.1, 656.5))
    
    # Axes labels
    plt.ylabel("Flux", c="white", fontsize=14)
    plt.xlabel("Wavelength (nm)", c="white", fontsize=14)

    
    # Title and subtitle
    subtitle = "RA: " + stardata[2] + "    DEC: " + stardata[3] + "    Spectral Type: " + stardata[5] + "    Candidate Type: " + stardata[6] + "    Young Association: " + stardata[7]
    ax1.title.set_text(subtitle)
    plt.suptitle(stardata[4],fontsize=24)
    
    # Save plot
    plottitle = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/H-alpha_Li Spectrum Plots/" + title + "HaLispectrumplot"
    plt.savefig(plottitle)
    
    plt.close()
    
    return



#--------------------------------------



def dopplershiftplot(wavelength, flux, stardata, title, amp):
    """
    This function creates a plot of the spectrum provided in the FITS file,
    cropped to the relevant region for Lithium and H-alpha absorption and doppler shifted by gaussian fit.
    
    Inputs:
    wavelength     Wavelength in nm
    flux           Outputted flux
    stardata       Array with filename, star identifier, RA, DEC, spectral type, candidate type, and young association
    title          Title to name saved figure
    amp            Amplitude estimate to be passed to Gaussian fit.
    
    Output:
    None
    
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

    #print(report) 
    
    
    values = []
    for param in result.params.values():
        #print(param.value)
        values.append(param.value)
    
    cont = values[0]
    amp = values[1]
    cend = values[2]
    wid = values[3]
    cen = 656.28

    y = cont + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(smthwvlen-cen)**2 / (2*wid**2))

    wvlenshift = cen - cend
    wvlens = wvlenshift + smthwvlen
    c = 2.99792e5
    ds = -np.log(wvlens/smthwvlen)*c
    #print(ds)
    #newwvlen = smthwvlen - wvlenshift
    newwvlen = smthwvlen*np.exp(-ds/c)

    # Figure size
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,15))
    
    # Plot the Gassian matched and shifted H-alpha
    ax1.plot(smthwvlen, smthflx, 'bo')
    ax1.plot(newwvlen, smthflx, "go", label="DS data")
    ax1.plot(smthwvlen, result.init_fit, 'k--', label='initial fit')
    ax1.plot(smthwvlen, result.best_fit, 'r-', label='best fit')
    ax1.plot(smthwvlen, y, 'c-', label="aligned curve")
    ax1.axvline(x=cen, color="navy", linestyle="dashed", label="H-alpha center")
    ax1.legend(loc='upper right')
    
    # Crop data for Li range
    li_lambcrop = np.where((wavelength > 669) & (wavelength < 672))
    # And make sure it is the same length as the H-alpha data
    liwv = wavelength[li_lambcrop]
    liwv = liwv[:len(ds)]
    liflx = flux[li_lambcrop]
    liflx = liflx[:len(ds)]
    shiftliwv = liwv*np.exp(-ds/c)

    # Plot li data
    ax2.plot(liwv, liflx, "r", label = 'unshifted data')
    ax2.plot(shiftliwv, liflx, 'c', label = "shifted data")
    ax2.axvline(x=670.78, color="k", linestyle="dashed", label = "Lithium center")
    ax2.legend(loc='upper right')
    ax2.set_xlim((670.58, 670.98))
    
    # Axes labels
    plt.ylabel("Flux", c="white", fontsize=14)
    plt.xlabel("Wavelength (nm)", c="white", fontsize=14)

    
    # Title and subtitle
    subtitle = "RA: " + stardata[2] + "    DEC: " + stardata[3] + "    Spectral Type: " + stardata[5] + "    Candidate Type: " + stardata[6] + "    Young Association: " + stardata[7]
    ax1.title.set_text(subtitle)
    plt.suptitle(stardata[4],fontsize=24)
    
    # Save plot
    plottitle = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/NewDopplerShiftPlots/" + title + "Dopplershiftplot"
    plt.savefig(plottitle)
    
    plt.close()
    
    return
    

#--------------------------------------



def plot_ew(wavelength, flux, error, stardata, title, amp):
    """
    Function which takes FITS files and calculates the Li equivalent widths. 
    Saves a figure and returns an array containing file name and equivalent widths.

    Input:
    wavelength     Wavelength in nm
    flux           Outputted flux
    error          Error of the flux
    stardata       Array with filename, star identifier, RA, DEC, spectral type, candidate type, and young association
    title          Title to name saved figure
    amp            Amplitude estimate to be passed to Gaussian fit

    Output: 
    None
    
    """

    liwvlen, liflx, lierr = lidopplershift(wavelength, flux, error, amp)

    gmodel = Model(slopedgaussian)

    # do fit
    result = gmodel.fit(liflx, x=liwvlen, slope=0, cont=0.6, amp=Parameter("amp", value=-0.3, max=1e-5), cen=Parameter("cen", value=670.78, min=670.7, max=670.85), wid=Parameter("wid", value=0.08, min=1e-2, max=0.09))


    #print(report)
    plt.clf()
    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Take values of fitted Gaussian
    values = []
    for param in result.params.values():
        #print(param.value)
        values.append(param.value)

    slope = values[0]
    cont = values[1]
    amp = values[2]
    cen = values[4]
    wid = values[3]


    # 2 curves to calculate area between (area of the Gaussian)
    gauss = lambda x: slope*x + cont + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

    continuum = lambda x: slope*x + cont

    x = np.linspace(670, 671.5, 500)

    fx = gauss(x)
    gx = continuum(x)

    plt.plot(x, fx, x, gx)
    plt.plot(liwvlen, liflx, "k")
    plt.fill_between(x, fx, gx)
    plt.xlim((670, 671.25))

    # Set axes colours
    ax.tick_params(which="major", size=7, labelsize=18, width=3)
    
    # Axes widths
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    plt.xlabel("Wavelength (nm)", fontsize=16)
    plt.ylabel("Flux", fontsize=16)

    fdu = integrate.quad(gauss, 670, 671.5)[0]
    gdu = integrate.quad(continuum, 670, 671.5)[0]

    area = gdu - fdu

    rng = np.where((gx-fx)> 9e-3)[0]
    if len(rng) == 0:
        plottitle = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/LiEW/" + title
    else:
        start = rng[0]
        end = rng[-1]

        # Solving for equivalent width using area of trapezoid
        a = gx[start]
        b = gx[end]

        ew = 2*area/(a+b)
        ewma = ew*10000 

        if ewma < 1000:

            c = cen - ew/2
            d = cen + ew/2

            plt.axvline(x = c, c="k", linestyle="dashed", label='EW = %.3f $m\AA$' %ewma)
            plt.axvline(x = d, c="k", linestyle="dashed")
            plt.legend(fontsize=14)
            plt.fill_between(x, fx, gx, where=(x > c) & (x < d), color="navy")
            plottitle = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/LiEW/" + title

    # Title and subtitle
    subtitle = "RA: " + stardata[2] + "    DEC: " + stardata[3] + "    Spectral Type: " + stardata[5] + "    Candidate Type: " + stardata[6] + "    Young Association: " + stardata[7]
    plt.title(subtitle,fontsize=16, c="k")
    plt.suptitle(stardata[4],fontsize=24, c="k")
            

    plt.savefig(plottitle)
    #plt.close()

    return 



#--------------------------------------


