from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
import os
import csv
import time
from ew_functions import *
from Li_plotting import *
from create_Li_GRP_fig import *


#--------------------------------------


def main():
    """
    Main function
    """
    # Counting code time
    tic = time.perf_counter()

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
    gaiaid = df.DR2_Source_ID.tolist()
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
    csvarray = np.empty((0, 9), str)

    for i in range(len(shortnames_csv)):
        if shortnames_csv[i] in fitsarray[:,1]:
            theindex = np.where(fitsarray[:,1] == shortnames_csv[i])
            betterindex = theindex[0]
            # Handles duplicates and appends to the array
            if len(betterindex) == 2:
                csvarray = np.append(csvarray, np.array([[np.str(fitsarray[betterindex[0],0]), np.str(fitsarray[betterindex[0],1]), np.str(fitsarray[betterindex[0],2]), np.str(fitsarray[betterindex[0],3]), fullnames[i], gaiaid[i], spectraltypes[i], ctypes[i], groups[i]]]), axis=0)
                csvarray = np.append(csvarray, np.array([[np.str(fitsarray[betterindex[1],0]), np.str(fitsarray[betterindex[1],1]), np.str(fitsarray[betterindex[1],2]), np.str(fitsarray[betterindex[1],3]), fullnames[i], gaiaid[i], spectraltypes[i], ctypes[i], groups[i]]]), axis=0)
            else:
                csvarray = np.append(csvarray, np.array([[np.str(fitsarray[theindex,0]), np.str(fitsarray[theindex,1]), np.str(fitsarray[theindex,2]), np.str(fitsarray[theindex,3]), fullnames[i], gaiaid[i], spectraltypes[i], ctypes[i], groups[i]]]), axis=0)

    # fixing the formatting for the fitsarray entries
    for j in range(len(csvarray[:,0])):
        csvarray[j,0] = csvarray[j,0].replace("[" ,"").replace("]","").replace("'", "")
        csvarray[j,1] = csvarray[j,1].replace("[" ,"").replace("]","").replace("'", "")
        csvarray[j,2] = csvarray[j,2].replace("[" ,"").replace("]","").replace("'", "")
        csvarray[j,3] = csvarray[j,3].replace("[" ,"").replace("]","").replace("'", "")

    # Lists for opening the file and plotting (strip fits)  
    ew_values = np.empty((0, 4), float)
    plotentries = []

    for row in range(len(csvarray[:,0])):
        # Strip .fits.gz for plotting
        file = csvarray[row, 0]
        #filelist.append(file) 
        plotfile = csvarray[row,0].strip(".fits.gz") + "i"  
        plotentries.append(plotfile)

        # Return wavelength and flux from ifile
        wavelength, flux, error, hdr = readfits(file) 
        liwvlen, liflx, lierr = lidopplershift(wavelength, flux, error, -0.1)
        params = gaussianfit(liwvlen, liflx)
        ew = calculate_ew(liwvlen, liflx, params)
        print(ew)
        ewstdev, ewmabsdev, nannumb = ewerror(file, "liew", -0.1, 2500)

        ew_values = np.append(ew_values, np.array([[ew, ewstdev, ewmabsdev, nannumb]]), axis=0)

        # Plot
        # fullspectrumplot(wavelength, flux, error, csvarray[row,:], plotfile)
        # Lispectrumplot(wavelength, flux, error, csvarray[row,:], plotfile)
        # h_alpha_liplot(wavelength, flux, error, csvarray[row,:], plotfile)
        # dopplershiftplot(wavelength, flux, csvarray[row,:], plotfile, -0.1)   
    
    csvarray = np.append(csvarray, ew_values, axis=1)

    # Write CSV of data
    header = ["File Name", "Short Name", "RA", "DEC", "Simbad ID", "Gaia DR2 ID", "Spectral Type", "Candidate Type", "Group", "EW", "EW St. Dev.", "EW Med. Abs. Dev.", "# of NaNs"]
    
    with open('Espadons_Li_EW.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(csvarray)

    # Finish counting time
    toc = time.perf_counter()
    print(f"This code took {toc - tic:0.4f} seconds.")


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

if __name__ == "__main__":
    main()
