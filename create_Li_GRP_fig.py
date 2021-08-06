from sqlalchemy import create_engine
import pandas as pd
from random import seed
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
from matplotlib.ticker import MultipleLocator



#--------------------------------------


def nyadb_query(query):
    """
    Queries the Nearby Young Association Database.

    Input:
    query         Command to query the database.

    Output:
    data          The result of the query as a Pandas Dataframe.
    """


    nyadb_file = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/nyadb_apr_2021_snapshot.db"
    #Temporary file for communication of data
    comm_dir = '/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/tmp_comm_files/'
    seed(1)
    comm_file = ""
    for i in range(4):
        value = str(gauss(0, 1))
        new_val = value.replace(".", "")
        new_val = new_val.replace("-", "")
        comm_file += new_val
    comm_file += '.csv'


    engine = create_engine('sqlite:///'+nyadb_file, echo=False)
    df = pd.read_sql_query(query, engine)
    df.to_csv(comm_dir+comm_file,index=False,na_rep='NaN',encoding='utf-8')
    
    data = pd.read_csv(comm_dir+comm_file, dtype = 'object')
    
    return data


#--------------------------------------


def background_stars(age_sort = True):
    """
    Queries the NYADB for stars with known equivalent widths to be plotted for reference.
    Input:
    age_sort        indicates if the function should return limits sorted into age values or not

    Outputs:
    xs              array of lists with all G - RP values sorted into age categories
    ys              array of lists with all EW values sorted into age categories
------
    xslim           array of lists with all G - RP upper limits sorted into age categories
    yslim           array of lists with all EW upper limits sorted into age categories

    OR 

    xlim            array of lists with all G - RP upper limits 
    ylim            array of lists with all EW upper limits
    

    """

    li_data = nyadb_query("SELECT g2.phot_g_mean_mag, g2.phot_rp_mean_mag, ew.*, ass.adopted_age_val, so.main_membership, bs.ya_prob FROM sources AS so INNER JOIN gaiadr2 as g2 ON CAST(g2.source_id AS TEXT)=so.gaiadr2_source_id INNER JOIN equivalent_widths AS ew ON (ew.source_id=so.id AND ew.ew_species='li') INNER JOIN banyan_sigma as bs ON (bs.source_id = so.id) INNER JOIN associations AS ass ON (ass.shortname = bs.best_ya) WHERE (bs.ya_prob > 0.9) AND bs.id IN (SELECT MIN(id) FROM banyan_sigma GROUP BY source_id)")

    #Fix the different notations for upper limits
    gg_lim = np.where(((li_data["ew_angstrom"] != 0) | (li_data['ew_angstrom_unc'] != 0)) & (li_data["ew_flag"] == 'upper limit') & (li_data["spectral_resolving_power"].astype(float) > 1e4) & (li_data["ya_prob"].astype(float) >= .9), )

    x = li_data['phot_g_mean_mag'].to_numpy(float) - li_data['phot_rp_mean_mag'].to_numpy(float)
    y = li_data['ew_angstrom'].to_numpy(float)*1e3
    age = li_data['adopted_age_val'].to_numpy(float)
    ey = li_data['ew_angstrom_unc'].to_numpy(float)*1e3


    ylim = y[gg_lim]
    eylim = ey[gg_lim]
    xlim = x[gg_lim]
    agelim = age[gg_lim]

    for i in gg_lim[0]:
        if (y[i] == 0) & (ey[i] != 0):
            y[i] = ey[i]

    for i in range(len(ylim)):
        if (ylim[i] == 0) & (eylim[i] != 0):
            ylim[i] = eylim[i]


    x20myr = []
    y20myr = []
    x25myr = []
    y25myr = []
    x40myr = []
    y40myr = []
    x75myr = []
    y75myr = []
    x150myr = []
    y150myr = []
    x200myr = []
    y200myr = []
    xlim20myr = []
    ylim20myr = []
    xlim25myr = []
    ylim25myr = []
    xlim40myr = []
    ylim40myr = []
    xlim75myr = []
    ylim75myr = []
    xlim150myr = []
    ylim150myr = []
    xlim200myr = []
    ylim200myr = []


    xs = []
    ys = []
    xslim = []
    yslim = []

    for i in range(len(x)):
        if age[i] < 20:
            x20myr.append(x[i])
            y20myr.append(y[i])
        elif (age[i] >= 20) and (age[i] < 30):
            x25myr.append(x[i])
            y25myr.append(y[i])
        elif (age[i] >= 30) and (age[i] < 50):
            x40myr.append(x[i])
            y40myr.append(y[i])
        elif (age[i] >= 50) and (age[i] < 100):
            x75myr.append(x[i])
            y75myr.append(y[i])
        elif (age[i] >= 100) and (age[i] < 200):
            x150myr.append(x[i])
            y150myr.append(y[i])
        elif age[i] >= 200:
            x200myr.append(x[i])
            y200myr.append(y[i])

            
    for i in range(len(xlim)):
        if agelim[i] < 20:
            xlim20myr.append(xlim[i])
            ylim20myr.append(ylim[i])
        elif (agelim[i] >= 20) and (agelim[i] < 30):
            xlim25myr.append(xlim[i])
            ylim25myr.append(ylim[i])
        elif (agelim[i] >= 30) and (agelim[i] < 50):
            xlim40myr.append(xlim[i])
            ylim40myr.append(ylim[i])
        elif (agelim[i] >= 50) and (agelim[i] < 100):
            xlim75myr.append(xlim[i])
            ylim75myr.append(ylim[i])
        elif (agelim[i] >= 100) and (agelim[i] < 200):
            xlim150myr.append(xlim[i])
            ylim150myr.append(ylim[i])
        elif agelim[i] >= 200:
            xlim200myr.append(xlim[i])
            ylim200myr.append(ylim[i])        

    xs.append(list(x20myr))
    xs.append(list(x25myr))
    xs.append(list(x40myr))
    xs.append(list(x75myr))
    xs.append(list(x150myr))
    xs.append(list(x200myr))
    ys.append(list(y20myr))
    ys.append(list(y25myr))
    ys.append(list(y40myr))
    ys.append(list(y75myr))
    ys.append(list(y150myr))
    ys.append(list(y200myr))
    xslim.append(list(xlim20myr))
    xslim.append(list(xlim25myr))
    xslim.append(list(xlim40myr))
    xslim.append(list(xlim75myr))
    xslim.append(list(xlim150myr))
    xslim.append(list(xlim200myr))
    yslim.append(list(ylim20myr))
    yslim.append(list(ylim25myr))
    yslim.append(list(ylim40myr))
    yslim.append(list(ylim75myr))
    yslim.append(list(ylim150myr))
    yslim.append(list(ylim200myr))

    if age_sort == True:
        return xs, ys, xslim, yslim 
    else:
        return xs, ys, xlim, ylim



#--------------------------------------

def overplotteddata(age_sort=True):
    """
    Queries the NYADB for stars with known equivalent widths to be plotted for reference.
    Input:
    age_sort        indicates if the function should return limits sorted into age values or not

    Outputs:
    g_rps           array of lists with all G - RP values sorted into age categories
    ews             array of lists with all EW values sorted into age categories
    g_rpups         array of lists with all G - RP upper limits sorted into age categories
    ewups           array of lists with all EW upper limits sorted into age categories

----- OR ------

    g_rp            array of lists with all G - RP values
    ew              array of lists with all EW values 
    g_rpup          array of lists with all G - RP upper limits 
    ewup            array of lists with all EW upper limits
    """

    liewdata = np.loadtxt("/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/Espadons_Li_EW-2.csv", dtype=str, delimiter=",", skiprows=1)
    gmag = np.array(liewdata[:, 6]).astype("float64")
    rpmag = np.array(liewdata[:, 7]).astype("float64")
    ew = np.array(liewdata[:, 11]).astype("float64")
    gaiaid = np.array(liewdata[:, 5]).astype("float64")

    ewerr = np.array(liewdata[:, 13]).astype("float64")
    g_rperr = np.zeros(len(gmag))

    nans = np.array(liewdata[:, 14]).astype("float64")

    g_rp = gmag - rpmag

    ewup = np.array([])
    g_rpup = np.array([])

    for i in range(len(ew)):
        if (nans[i] > 0) and (nans[i] < 2500):
            uplim = ew[i] + ewerr[i]
            ew[i] = ew[i] + ewerr[i]
            ewerr[i] = 0
            ewup = np.append(ewup, uplim)
            g_rpup = np.append(g_rpup, g_rp[i])

    alldat = nyadb_query("SELECT * FROM sources LEFT OUTER JOIN associations ON (associations.shortname=sources.main_membership)")

    age = alldat["adopted_age_val"].to_numpy("float64")
    datgaiaid = alldat["gaiadr2_source_id"].to_numpy("float64")
    g_rpage = np.array([])
    ewage = np.array([])
    thage = np.array([])
    ewerrage = np.array([])
    nanage = np.array([])

    for i in range(len(datgaiaid)):
        for j in range(len(gaiaid)):
            if datgaiaid[i] == gaiaid[j]:
                g_rpage = np.append(g_rpage, g_rp[j])
                ewage = np.append(ewage, ew[j])
                ewerrage = np.append(ewerrage, ewerr[j])
                nanage = np.append(nanage, nans[j])
                thage = np.append(thage, age[i])

    ewupage = np.array([])
    g_rpupage = np.array([])
    upage = np.array([])

    for i in range(len(ewage)):
        if (nanage[i] > 0) and (nanage[i] < 2500):
            uplimage = ewage[i] + ewerrage[i]
            ewupage = np.append(ewupage, uplimage)
            g_rpupage = np.append(g_rpupage, g_rpage[i])
            upage = np.append(upage, thage[i])
            
    
    x20myr = []
    y20myr = []
    x25myr = []
    y25myr = []
    x40myr = []
    y40myr = []
    x75myr = []
    y75myr = []
    x150myr = []
    y150myr = []
    x200myr = []
    y200myr = []
    yerr20myr = []
    yerr25myr = []
    yerr40myr = []
    yerr75myr = []
    yerr150myr = []
    yerr200myr = []
    ewup20myr = []
    g_rpup20myr = []
    ewup25myr = []
    g_rpup25myr = []
    ewup40myr = []
    g_rpup40myr = []
    ewup75myr = []
    g_rpup75myr = []
    ewup150myr = []
    g_rpup150myr = []
    ewup200myr = []
    g_rpup200myr = []

    xnan = []
    ynan = []
    yerrnan = []
    g_rpupnan = []
    ewupnan = []


    g_rps = []
    ews = []
    ewserr = []
    ewups = []
    g_rpups = []

    nanages = []

    for i in range(len(g_rpage)):
        if thage[i] < 20:
            x20myr.append(g_rpage[i])
            y20myr.append(ewage[i])
            yerr20myr.append(ewerrage[i])
        elif (thage[i] >= 20) and (thage[i] < 30):
            x25myr.append(g_rpage[i])
            y25myr.append(ewage[i])
            yerr25myr.append(ewerrage[i])
        elif (thage[i] >= 30) and (thage[i] < 50):
            x40myr.append(g_rpage[i])
            y40myr.append(ewage[i])
            yerr40myr.append(ewerrage[i])
        elif (thage[i] >= 50) and (thage[i] < 100):
            x75myr.append(g_rpage[i])
            y75myr.append(ewage[i])
            yerr75myr.append(ewerrage[i])
        elif (thage[i] >= 100) and (thage[i] < 200):
            x150myr.append(g_rpage[i])
            y150myr.append(ewage[i])
            yerr150myr.append(ewerrage[i])
        elif thage[i] >= 200:
            x200myr.append(g_rpage[i])
            y200myr.append(ewage[i])
            yerr200myr.append(ewerrage[i])
        else:
            xnan.append(g_rpage[i])
            ynan.append(ewage[i])
            yerrnan.append(ewerrage[i])
            nanages.append(thage[i])

    for i in range(len(g_rpupage)):
        if upage[i] < 20:
            g_rpup20myr.append(g_rpupage[i])
            ewup20myr.append(ewupage[i])
        elif (upage[i] >= 20) and (upage[i] < 30):
            g_rpup25myr.append(g_rpupage[i])
            ewup25myr.append(ewupage[i])
        elif (upage[i] >= 30) and (upage[i] < 50):
            g_rpup40myr.append(g_rpupage[i])
            ewup40myr.append(ewupage[i])
        elif (upage[i] >= 50) and (upage[i] < 100):
            g_rpup75myr.append(g_rpupage[i])
            ewup75myr.append(ewupage[i])
        elif (upage[i] >= 100) and (upage[i] < 200):
            g_rpup150myr.append(g_rpupage[i])
            ewup150myr.append(ewupage[i])
        elif upage[i] >= 200:
            g_rpup200myr.append(g_rpupage[i])
            ewup200myr.append(ewupage[i])
        else:
            g_rpupnan.append(g_rpupage[i])
            ewupnan.append(ewupage[i])
            

    g_rps.append(list(x20myr))
    g_rps.append(list(x25myr))
    g_rps.append(list(x40myr))
    g_rps.append(list(x75myr))
    g_rps.append(list(x150myr))
    g_rps.append(list(x200myr))
    g_rps.append(list(xnan))
    ews.append(list(y20myr))
    ews.append(list(y25myr))
    ews.append(list(y40myr))
    ews.append(list(y75myr))
    ews.append(list(y150myr))
    ews.append(list(y200myr))
    ews.append(list(ynan))
    ewserr.append(list(yerr20myr))
    ewserr.append(list(yerr25myr))
    ewserr.append(list(yerr40myr))
    ewserr.append(list(yerr75myr))
    ewserr.append(list(yerr150myr))
    ewserr.append(list(yerr200myr))
    ewserr.append(list(yerrnan))

    ewups.append(list(ewup20myr))
    g_rpups.append(list(g_rpup20myr))
    ewups.append(list(ewup25myr))
    g_rpups.append(list(g_rpup25myr))
    ewups.append(list(ewup40myr))
    g_rpups.append(list(g_rpup40myr))
    ewups.append(list(ewup75myr))
    g_rpups.append(list(g_rpup75myr))
    ewups.append(list(ewup150myr))
    g_rpups.append(list(g_rpup150myr))
    ewups.append(list(ewup200myr))
    g_rpups.append(list(g_rpup200myr))
    ewups.append(list(ewupnan))
    g_rpups.append(list(g_rpupnan))

    g_rpserr = np.array([np.zeros(len(x20myr)), np.zeros(len(x25myr)), np.zeros(len(x40myr)), np.zeros(len(x75myr)), np.zeros(len(x150myr)),np.zeros(len(x200myr)), np.zeros(len(xnan))])
  
    if age_sort == False:
        return ew, g_rp, ewerr, g_rperr, ewup, g_rpup 

    else:
        return ews, g_rps, ewserr, g_rpserr, ewups, g_rpups

#--------------------------------------

def gaiabackgroundplot():
    """
    Creates a plot with all of the background stars and overplotted data not sorted into age categories.

    """
    xs, ys, xlim, ylim = background_stars(age_sort = False)
    ew, g_rp, ewerr, g_rperr, ewup, g_rpup = overplotteddata(age_sort = False)

    fig = plt.figure(figsize=(15, 15))
    ax = plt.gca()
        
    age_ranges = [[0.,20.], [20.,30.], [30.,50.], [50.,100.], [100.,200.], [200.,700.]]
    nage = len(age_ranges)

    edgecolors = ["indigo", "royalblue", "springgreen", "gold", "darkorange", "red"]
    symbols = ["o", "^", "<", ">", "v", "d"]
    symbol_sizes = [41, 50, 50, 50, 50, 50]
    labels = ["< 20 Myr", "20-30 Myr", "30-50 Myr", "50-100 Myr", "100-200 Myr", "$\geq$ 200 Myr"]

    for i in range(nage):
        plt.scatter(xs[i], ys[i], color='white', edgecolors = edgecolors[i], marker=symbols[i], s=symbol_sizes[i], linewidths=3, alpha=0.7, label=labels[i])

        
    plt.errorbar(g_rp, ew, ewerr, g_rperr, color='white', ecolor ='grey', marker='*', ls='none', mec='k', ms=18, mew=3, alpha = 0.8, label = "Espadons Data")
        
    arrow = u'$\u2193$'
    ax.plot(xlim-0.003, ylim-12, linestyle='none', marker=arrow, c= "dimgrey", markersize=15, alpha = 0.7)
    ax.plot(g_rpup-0.003, ewup-12, linestyle='none', marker=arrow, c= "dimgrey", markersize=15, alpha = 0.7)



    plt.legend(loc="upper left", prop={'size': 25})    

    xB = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    x = [0.230, 0.56, 0.92, 1.33]
    spec = ["F0", "K0", "M0", "M5"]
    axT = ax.secondary_xaxis('top')

    axT.set_xticks(x)
    axT.set_xticklabels(spec)

    ax.set_xticks(xB)

    # Minor axes
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    axT.xaxis.set_minor_locator(MultipleLocator(0.05))

    # Set axes colours
    ax.tick_params(which="major", size=9, labelsize=21, width=3, direction='in',  right=True)
    ax.tick_params(which="minor", size=5, labelsize=12, width=2, direction='in',  right=True)
    axT.tick_params(which="major", size=9, labelsize=21, width=3, direction='in')
    axT.tick_params(which="minor", size=5, labelsize=12, width=2, direction='in')



    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'

    # ax.set_xticklabels(labels)

    # Axes widths
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    ax.set_xlabel("Gaia DR2 G-G$_{RP}$", fontsize=25)
    axT.set_xlabel("Spectral Type", fontsize=25)
    ax.set_ylabel("Li I EW 6707.7$\AA$ ($m\AA$)", fontsize=25)

    plt.ylim((0, 800))
    plt.xlim((0.1, 1.5))
    plt.savefig("/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/EWGRPfigs/LiGaiaplotbgrnd")

    return



#--------------------------------------


def onlyoverplot():
    """
    Creates a plot with only the ESPaDOns overplotted data, sorted into age categories.
    """
    ews, g_rps, ewserr, g_rpserr, ewups, g_rpups = overplotteddata()
    arrow = u'$\u2193$'

    fig = plt.figure(figsize=(15, 15))
    ax = plt.gca()

    age_ranges = [[0.,20.], [20.,30.], [30.,50.], [50.,100.], [100.,200.], [200.,700.], ["nan"]]
    nage = len(age_ranges)

    edgecolors = ["indigo", "royalblue", "springgreen", "gold", "darkorange", "red", "dimgrey"]
    labels = ["< 20 Myr", "20-30 Myr", "30-50 Myr", "50-100 Myr", "100-200 Myr", "$\geq$ 200 Myr", "No age provided"]

    for i in range(nage):
        plt.errorbar(g_rps[i], ews[i], ewserr[i], g_rpserr[i], color='white', ecolor ='grey', marker='*', ls='none', mec=edgecolors[i], ms=18, mew=3, alpha = 0.8, label = labels[i])
        ax.plot(np.array(g_rpups[i])-0.0035, np.array(ewups[i])-2, linestyle='none', marker=arrow, c= "dimgrey", markersize=25, alpha = 0.7)

    plt.legend(loc="upper left", prop={'size': 25})    

    xB = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    x = [0.230, 0.56, 0.92]
    spec = ["F0", "K0", "M0"]
    axT = ax.secondary_xaxis('top')

    axT.set_xticks(x)
    axT.set_xticklabels(spec)

    ax.set_xticks(xB)

    # Minor axes
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(4))
    axT.xaxis.set_minor_locator(MultipleLocator(0.05))

    # Set axes colours
    ax.tick_params(which="major", size=9, labelsize=21, width=3, direction='in',  right=True)
    ax.tick_params(which="minor", size=5, labelsize=12, width=2, direction='in',  right=True)
    axT.tick_params(which="major", size=9, labelsize=21, width=3, direction='in')
    axT.tick_params(which="minor", size=5, labelsize=12, width=2, direction='in')


    # Axes widths
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    ax.set_xlabel("Gaia DR2 G-G$_{RP}$", fontsize=25)
    axT.set_xlabel("Spectral Type", fontsize=25)
    ax.set_ylabel("Li I EW 6707.7$\AA$ ($m\AA$)", fontsize=25)

    plt.ylim((0, 175))
    plt.xlim((0.1, 1.1))
    plt.savefig("/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/EWGRPfigs/LiEWvsGRP")

    return


#--------------------------------------

def agedoverplotwback():
    """
    Creates plots with each of the age categories of overplotted stars and background stars.
    """
    
    xs, ys, xslim, yslim = background_stars()
    ews, g_rps, ewserr, g_rpserr, ewups, g_rpups = overplotteddata()

    age_ranges = [[0.,20.], [20.,30.], [30.,50.], [50.,100.], [100.,200.], [200.,700.]]
    nage = len(age_ranges)

    edgecolors = ["indigo", "royalblue", "lime", "gold", "darkorange", "red"]
    symbols = ["o", "^", "<", ">", "v", "d"]
    symbol_sizes = [41, 50, 50, 50, 50, 50]
    labels = ["< 20 Myr", "20-30 Myr", "30-50 Myr", "50-100 Myr", "100-200 Myr", "$\geq$ 200 Myr"]
    agename = ["lt_20Myr", "20-30Myr", "30-50Myr", "50-100Myr", "100-200_Myr", "geth200_Myr"]

    for i in range(nage):
        fig = plt.figure(figsize=(15, 15))
        ax = plt.gca()

        for j in range(nage):
            if j != i:
                plt.scatter(xs[j], ys[j], color='white', edgecolors = edgecolors[j], marker=symbols[j], s=symbol_sizes[j], linewidths=3, alpha=0.2, label=labels[j])    
        
        plt.scatter(xs[i], ys[i], color='white', edgecolors = edgecolors[i], marker=symbols[i], s=symbol_sizes[i], linewidths=3, alpha=0.8, label=labels[i])    
        plt.errorbar(g_rps[i], ews[i], ewserr[i], g_rpserr[i], color='white', ecolor ='grey', marker='*', ls='none', mec="k", ms=18, mew=3, alpha = 0.9, label = "Espadons Data")

        arrow = u'$\u2193$'
        ax.plot(np.array(xslim[i])-0.003, np.array(yslim[i])-12, linestyle='none', marker=arrow, c= "dimgrey", markersize=15, alpha = 0.7)
        ax.plot(np.array(g_rpups[i])-0.003, np.array(ewups[i])-12, linestyle='none', marker=arrow, c= "dimgrey", markersize=15, alpha = 0.7)

        
        plt.legend(loc="upper left", prop={'size': 25})    

        xB = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        xT = [0.230, 0.56, 0.92, 1.33]
        spec = ["F0", "K0", "M0", "M5"]
        axT = ax.secondary_xaxis('top')

        axT.set_xticks(xT)
        axT.set_xticklabels(spec)

        ax.set_xticks(xB)

        # Minor axes
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        axT.xaxis.set_minor_locator(MultipleLocator(0.05))

        # Set axes colours
        ax.tick_params(which="major", size=9, labelsize=21, width=3, direction='in',  right=True)
        ax.tick_params(which="minor", size=5, labelsize=12, width=2, direction='in',  right=True)
        axT.tick_params(which="major", size=9, labelsize=21, width=3, direction='in')
        axT.tick_params(which="minor", size=5, labelsize=12, width=2, direction='in')


        # Axes widths
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)

        ax.set_xlabel("Gaia DR2 G-G$_{RP}$", fontsize=25)
        axT.set_xlabel("Spectral Type", fontsize=25)
        ax.set_ylabel("Li I EW 6707.7$\AA$ ($m\AA$)", fontsize=25)

        plt.ylim((0, 800))
        plt.xlim((0.1, 1.5))
        title = "/Users/samantha/OneDrive - The University of Western Ontario/Research Summer 2021/EWGRPfigs/EWvsGRP" + agename[i]
        plt.savefig(title)

        return

#--------------------------------------