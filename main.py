from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
from astropy import units as u
from astropy.visualization import quantity_support
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from astropy.io import ascii
from scipy import optimize
from lmfit import Model
from lmfit.model import save_modelresult
from shutil import move
from tempfile import NamedTemporaryFile
import scipy.integrate as integrate

#--------------------------------------


def main():
    """
    Main function
    """
    # Make a list of the files in the 18BC22_raw_espadons folder
    entries = os.listdir("/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/18BC22_raw_espadons")

    # Lists for opening the file and plotting (strip fits)  
    ientries = []

    # Iterate over files in the folder 
    for file in entries:
        
        # Only want reduced files ending in i
        if file.endswith("i.fits.gz"):
            ientries.append(file) # append files ending in i


    # Make array of csv file to read
    df = pd.read_csv("/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/TGAS_BANYAN_SIGMA - Feb2018.csv",low_memory=False,skiprows=[0,2],dtype='object',na_values='...')

    # important information from csv for later
    shortnames_csv = df.shortname.tolist()
    fullnames = df.simbad_id_main.tolist()
    spectraltypes = df.spt_main.tolist()
    ctypes = df.type_main.tolist()
    groups = df.group_main.tolist()

    # array to hold fits file name and header info (objname, objra, and objdec)
    fitsarray = np.empty((0, 4), str)

    # loop to compose fitsarray
    for ifile in ientries:
        
        title = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/18BC22_raw_espadons/" + ifile 
        
        with fits.open(title) as hdu:
            hdr = hdu[0].header # extract the header
            objname = hdr["OBJNAME"]
            objra = hdr["OBJRA"]
            objdec = hdr["OBJDEC"]
            
        
        # omit the FJ objnames and strip the J's
        if objname.startswith("J"):
            shrt_name = objname.strip("J")
            fitsarray = np.append(fitsarray, np.array([[ifile, shrt_name, objra, objdec]]), axis=0)
        
    # array to hold all relevant information from fits and csv file (fitsarray, full name, spect. type, candidate, group)
    csvarray = np.empty((0, 8), str)

    for i in range(len(shortnames_csv)):
        if shortnames_csv[i] in fitsarray[:,1]:
            theindex = np.where(fitsarray[:,1] == shortnames_csv[i])
            # FOR NOW ONLY: REMOVING DUPICATES FROM LIST - will need to figure out the proper spectra for these later
            theindex = [i[0] for i in theindex]
            #print(theindex)
            csvarray = np.append(csvarray, np.array([[np.str(fitsarray[theindex,0]), np.str(fitsarray[theindex,1]), np.str(fitsarray[theindex,2]), np.str(fitsarray[theindex,3]), fullnames[i], spectraltypes[i], ctypes[i], groups[i]]]), axis=0)

    # fixing the formatting for the fitsarray entries
    for j in range(len(csvarray[:,0])):
        csvarray[j,0] = csvarray[j,0].replace("[" ,"").replace("]","").replace("'", "")
        csvarray[j,1] = csvarray[j,1].replace("[" ,"").replace("]","").replace("'", "")
        csvarray[j,2] = csvarray[j,2].replace("[" ,"").replace("]","").replace("'", "")
        csvarray[j,3] = csvarray[j,3].replace("[" ,"").replace("]","").replace("'", "")

    # Lists for opening the file and plotting (strip fits)  
    filelist = []
    plotentries = []

    for row in range(len(csvarray[:,0])):
        # Strip .fits.gz for plotting
        file = csvarray[row, 0]
        #filelist.append(file) 
        plotfile = csvarray[row,0].strip(".fits.gz") + "i"  
        plotentries.append(plotfile)
            
        # Return wavelength and flux from ifile
        # wavelength, flux, error, hdr = readfits(file) 
        # # Plot
        # #fullspectrumplot(wavelength, flux, error, csvarray[row,:], plotfile)
        # #Lispectrumplot(wavelength, flux, error, csvarray[row,:], plotfile)
        # #h_alpha_liplot(wavelength, flux, error, csvarray[row,:], plotfile)
        # dopplershift(wavelength, flux, csvarray[row,:], plotfile, -0.1)      

    #-----------------------------------------------

    # # Malo files for calculate_ew
    maloentrypath = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/reduced_Lison_Malo"
    maloplotpath = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/MaloEW/"

    # # Espadons files for calcultae_ew
    lientrypath = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/18BC22_raw_espadons"
    liplotpath = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/LiEW/"
    

    #maloew = calculate_ew(maloentrypath, maloplotpath, 0.5)
    liew = calculate_ew(lientrypath, liplotpath, -0.1)

    return 


#--------------------------------------


def readfits(filename):
    """
    This function takes the name of the FITS file to be read, 
    opens it, and extracts wavelength/flux/error data to be analysed
    
    Input:
    filename       The name of the file to be read
    
    Outputs:
    wavelength     Wavelength in nm
    flux           Outputted flux
    error          Error of the flux
    hdr            Header of the FITS file
    title          Title to name saved figure
    """
    
    title = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/18BC22_raw_espadons/" + filename 
    
    with fits.open(title) as hdu:
        data = hdu[0].data # extract array of data
        hdr = hdu[0].header # extract the header
    
    wavelength = data[0, :]
    flux = data[1, :]
    error = data[4, :]
    
    return wavelength, flux, error, hdr


#--------------------------------------


def fullspectrumplot(wavelength, flux, error, stardata, title):
    """
    This function creates a plot of the full spectrum provided in the FITS file.
    
    Inputs:
    wavelength     Wavelength in nm
    flux           Outputted flux
    error          Error of the flux
    stardata       Array with filename, star identifier, RA, DEC, spectral type, candidate type, and young association
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
    stardata       Array with filename, star identifier, RA, DEC, spectral type, candidate type, and young association
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
    stardata       Array with filename, star identifier, RA, DEC, spectral type, candidate type, and young association
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
    plottitle = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/H-alpha_Li Spectrum Plots/" + title + "Lispectrumplot"
    plt.savefig(plottitle)
    
    plt.close()
    
    return


#--------------------------------------


def gaussian(x, cont, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return cont + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

#--------------------------------------


def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w


#--------------------------------------


def dopplershift(wavelength, flux, stardata, title, amp):
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

def lidopplershift(wavelength, flux, amp):
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
        #print(param.value)
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
    shiftliwv = liwv*np.exp(-ds/c)

    return shiftliwv, liflx


#--------------------------------------

# Writing Malo files for pyEW
def pyewfiles(entrypath, ascpath, spectrapath, amp):
    """
    This function writes files for use in pyew from spectra.

    Inputs:
    entrypath           Full path to the folder in which the spectra are stored.
    ascpath             Full path + "/" to where the asc files will be stored.
    spectrapath         Full path to spectra.list.
    amp                 Amplitude estimate to be passed to the Gaussian fit. (0.5 for Malo, -0.1 for other)
    
    Output:
    None

    
    """

    entries = os.listdir(entrypath)

    # Lists for opening the file and plotting (strip fits)  
    ientries = []

    # Iterate over files in the folder 
    for file in entries:
        
        # Only want reduced files ending in i
        if file.endswith("i.fits.gz"): 
            ientries.append(file) # append files ending in i

    for ientry in ientries:
    
        title = entrypath + "/" + ientry
        file = ientry.strip('fits.gz')
        
        with fits.open(title) as hdu:
            data = hdu[0].data # extract array of data
            
        lamb = data[0, :]
        flux = data[1, :]

        wavelength, flux = lidopplershift(lamb, flux, amp)

        table = {'wavelength': wavelength, 'flux': flux}
    
        filename = ascpath + file + '.asc'
        ascii.write(table, filename, overwrite=True)
        
        with open(spectrapath, 'ab') as spectra:
            spectra.write((filename+'\n').encode("ascii"))
            
        temp_file = None
        
        with open(filename, 'r') as f_in:
            with NamedTemporaryFile(mode='w', delete=False) as f_out:
                temp_path = f_out.name
                next(f_in)
                for line in f_in:
                    f_out.write(line)
        
        os.remove(filename)
        move(temp_path, filename)     


#--------------------------------------


def slopedgaussian(x, slope, cont, amp, cen, wid):
    """1-d gaussian: gaussian(x, slope, cont, amp, cen, wid)"""
    return slope*x + cont + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


#--------------------------------------


def calculate_ew(filepath, plotpath, amp):
    """
    Function which takes FITS files and calculates the Li equivalent widths. 
    Saves a figure and returns an array containing file name and equivalent widths.

    Input:
    filepath              Full path to where FITS files are stored.
    plotpath              Full path + "/" to where plots are saved.

    Output: 
    ewarray               2D numpy array containing file name and equivalent width.
    
    """
    
    ewarray = np.empty((0, 2), float)

    files = os.listdir(filepath)
    ifiles = []

    # Iterate over files in the folder 
    for file in files:
        
        # Only want reduced files ending in i
        if file.endswith("i.fits.gz"): 
            ifiles.append(file) # append files ending in i


    for file in ifiles:
        title = filepath + '/' + file
        #print(file)

        with fits.open(title) as fitsfile:
            data = fitsfile[0].data
            hdr = fitsfile[0].header
        
        objname = hdr["OBJNAME"]
        lamb = data[0, :]
        flux = data[1, :]

        liwvlen, liflx = lidopplershift(lamb, flux, amp)

        gmodel = Model(slopedgaussian)
        result = gmodel.fit(liflx, x=liwvlen, slope=0, cont=0.6, amp=-0.2, cen=670.78, wid=0.08)

        
        # Take values of fitted Gaussian
        values = []
        for param in result.params.values():
            #print(param.value)
            values.append(param.value)

        slope = values[0]
        cont = values[1]
        amp = values[2]
        cen = values[3]
        wid = values[4]

        fwhm = 2*np.sqrt(2*np.log(2))*wid

        # 2 curves to calculate area between (area of the Gaussian)
        gauss = lambda x: slope*x + cont + (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

        continuum = lambda x: slope*x + cont

        x = np.linspace(670, 671.5, 500)

        fx = gauss(x)
        gx = continuum(x)

        plt.clf()
        plt.plot(x, fx, x, gx)
        plt.plot(liwvlen, liflx, "k")
        plt.fill_between(x, fx, gx)
        plt.xlim((670, 671.25))
        plt.title(objname)

        fdu = integrate.quad(gauss, 670, 671.5)[0]
        gdu = integrate.quad(continuum, 670, 671.5)[0]

        area = gdu - fdu

        rng = np.where((gx-fx)> 9e-3)[0]
        if (len(rng) == 0) or (cen > 670.85) or (cen < 670.7):
            #print("No Lithium present in %s." % file)
            plotname = plotpath + "ewfail" + file.strip(".fits.gz")
            plt.savefig(plotname)
            
        else:
            start = rng[0]
            end = rng[-1]

        #     plt.axvline(x = x[start], c="r", linestyle="dashed")
        #     plt.axvline(x = x[end], c="r", linestyle="dashed")

            # Solving for equivalent width using area of trapezoid
            a = gx[start]
            b = gx[end]

            ew = 2*area/(a+b)
            ewarray = np.append(ewarray, np.array([[file, ew]]), axis=0)

            c = cen - ew/2
            d = cen + ew/2

            plt.axvline(x = c, c="k", linestyle="dashed", label='EW = %.6f' %ew)
            plt.axvline(x = d, c="k", linestyle="dashed")
            plt.legend()

            plotname = plotpath + "ew" + file.strip(".fits.gz")
            plt.savefig(plotname)

    return ewarray


#--------------------------------------


if __name__ == "__main__":
    main()
