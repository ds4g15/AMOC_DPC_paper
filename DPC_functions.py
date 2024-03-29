import os
import time
import numpy as np
import xarray as xr
import xmitgcm as xm
import cartopy.feature
import ecco_v4_py as ecco
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.sparse import linalg as la
from matplotlib.colors import Normalize

GDS=xr.open_dataset('ECCO-GRID.nc') # Grid DataSet

# Get indices of wet points for 
atlmskC=ecco.get_basin_mask('atlExt',GDS.hFacC,less_output=True)
atlmskC=atlmskC.values[0,:]
atlmskC[((GDS.YC.values>80) | (GDS.YC.values<-35))]=0
Ti=np.where(atlmskC.flatten()>0)[0]
Tlen=len(Ti)
Si=Ti;

atlmskW=ecco.get_basin_mask('atlExt',GDS.hFacW,less_output=True)
atlmskW=atlmskW.values[0,:]
atlmskW[((GDS.YC.values>80) | (GDS.YC.values<-35))]=0
Ui=np.where(atlmskW.flatten()>0)[0]
Ulen=len(Ui)

atlmskS=ecco.get_basin_mask('atlExt',GDS.hFacS,less_output=True)
atlmskS=atlmskS.values[0,:]
atlmskS[((GDS.YG.values>80) | (GDS.YG.values<-35))]=0
Vi=np.where(atlmskS.flatten()>0)[0]
Vlen=len(Vi)

def unpack_subset_to_llc90(X):
    '''
    Return to the full ECCO grid surface data which has been flattened and subset.
    "X" is a 1- or 2-dimensional array whose first dimension 
                       should be the flattened spatial index 
                       (e.g. a space x time data matrix)
    Returns "Y", a 13x90x90 array.
    '''
    if   len(X)==Tlen: i=Ti
    elif len(X)==Ulen: i=Ui
    elif len(X)==Vlen: i=Vi
    if len(X)==Ulen+Vlen:
        if len(np.shape(X))==2: # Space x time
            nt=np.shape(X)[-1]  # number of time points
            Y1=np.zeros((13*90*90,nt))
            Y2=np.zeros((13*90*90,nt))
            Y1[Ui,:]=X[:Ulen,:]
            Y2[Vi,:]=X[Ulen:,:]
            Y=[Y1.reshape(13,90,90,nt),Y2.reshape(13,90,90,nt)] 
        else:                     # just space
            Y1=np.zeros(13*90*90)
            Y2=np.zeros(13*90*90)
            Y1[Ui]=X[:Ulen]
            Y2[Vi]=X[Ulen:]
            Y=[Y1.reshape(13,90,90),Y2.reshape(13,90,90)]
    else:
        if len(np.shape(X))==2:  # Space x time
            nt=np.shape(X)[-1]   # number of time points
            Y=np.zeros((13*90*90,nt))
            Y[i,:]=X[:,:]
            Y=Y.reshape(13,90,90,nt)            
        else:                    # just space
            Y=np.zeros(13*90*90) 
            Y[i]=X[:]
            Y=Y.reshape(13,90,90)                       
    return Y

def unpack_and_map(F,ax=plt.gca(),**kwargs):
    '''
    Quick plotting function for Atlantic area subset data.
    Take subset data, unpack it using "unpack_subset_to_llc90", 
    and pass this onto the "atl_map" function for plotting
    
    Parameters:
    -----------
    F:  1-dimensional ECCO data that has been flattened and subset. 
        If F contains concatenated zonal and meridional data, these
        are re-oriented and cast onto the C grid. Two subplots are
        returned, one zonal, one meridional.
    ax: AxesSubplot or List of AxesSubplots on which to create maps.
        If F contains concatenated zonal and meridional data, len(ax)
        should be 2.
    kwargs are ultimately passed to matplotlib.pyplot.pcolormesh for
    controlling plot appearance.
    '''
    if len(F)==Ulen+Vlen: # zonal/meridional velocity data
        if len(ax)!=2:
            raise ValueError('"ax" should be of length 2 for \
            concatenated zonal and meridional velocity data')
        FU,FV=F[:Ulen],F[Ulen:] # separate into u and v
        # expand onto LLC90 grid:
        FU,FV=unpack_subset_to_llc90(FU),unpack_subset_to_llc90(FV)

        # Reorient onto C grid
        FU=xr.DataArray(FU,dims=['tile','j','i_g'])
        FV=xr.DataArray(FV,dims=['tile','j_g','i'])
        FU,FV=ecco.vector_calc.UEVNfromUXVY(FU,FV,GDS)

        # Pass reoriented data to atl_map
        ax1,ax2=ax[:]
        p1,ax1=atl_map(GDS.XC,GDS.YC,FU,ax=ax1,**kwargs)
        p2,ax2=atl_map(GDS.XC,GDS.YC,FV,ax=ax2,**kwargs)        

        return p1,ax1,p2,ax2

    if   len(F)==Tlen: X,Y=GDS.XC,GDS.YC
    elif len(F)==Ulen: X,Y=GDS.XG,GDS.YC
    elif len(F)==Vlen: X,Y=GDS.XC,GDS.YG

    p,ax=atl_map(X,Y,unpack_subset_to_llc90(F),ax,**kwargs)
    
    return p,ax

def atl_map(lon,lat,F,ax=plt.gca(),**kwargs):
    ''' Creates a Lambert Azimuthal Equal Area map on axis centred on the Atlantic. 
        Note that the original axis is deleted and replaced with a new one in the same position
        with the correct projection, so for an array of subplots the original axis will need to
        be replaced with the new one in the array, e.g. p,AX=atl_map(lon,lat,F,ax=ax[0]);ax[0]=AX
        
        Returns:
        ---------
        p : a pcolormesh plot
        AX: the new GeoAxis
    '''
    latmin=-35
    latmax=90
    deltalat=0.25
    lonmin=-100
    lonmax=30
    deltalon=0.25

    lonc,latc,lone,late,FR=ecco.resample_to_latlon(lon, lat, F,latmin,latmax,deltalat,lonmin,lonmax,deltalon,mapping_method='nearest_neighbor',
                                   radius_of_influence = 112000)
    
    PROJ=ccrs.LambertAzimuthalEqualArea(central_latitude=0,central_longitude= -34.3)
    #### Create new axis
    POS=ax.get_position()
    FIG=ax.figure
    ax.remove()
    AX=FIG.add_axes(POS,projection=PROJ)
    #VLIM=np.max(np.abs(FR))
    p=AX.pcolormesh(lonc,latc,FR,transform=ccrs.PlateCarree(),zorder=-1,**kwargs)
    AX.coastlines(zorder=0,color='k')
    AX.add_feature(cartopy.feature.LAND, color='k')
    AX.set_xlim(np.array(AX.get_xlim())*0.85)
    AX.set_ylim(np.array(AX.get_ylim())*0.85)
    
    return p,AX

def atlstream(F,ax=plt.gca(),widthfactor=1,ss=1,vmin=None,vmax=None,**kwargs):
    '''
    Create a streamplot of flattened, subset, concatenated directional data.
    Parameters:
    -----------
    F: 1-dimensional array containing the data
    ax: AxesSubplot (will be deleted and replaced with a GeoAxes in the same position)
    widthfactor: float between 0 and 1. The width of the streamlines will be the local
                 magnitude (e.g. speed) divided by the maximum magnitude, multiplied by widthfactor
    ss (int): step used to step through data. Streamplots can be drawn more quickly if less data
                are used. The field that is plotted will be stepped through at a rate of "ss"
    vmin,vmax: colormap limits. Both the color and linewidth are a factor of the local magnitude.
                Defaults to the minimum and maximum latitudes
    Returns:
    --------
    s: streamplot object
    AX: GeoAxes replacing the supplied AxesSubplot
    '''    
    
    FU,FV=F[:Ulen],F[Ulen:]
    FU,FV=unpack_subset_to_llc90(FU),unpack_subset_to_llc90(FV)

    FU=xr.DataArray(FU,dims=['tile','j','i_g'])
    FV=xr.DataArray(FV,dims=['tile','j_g','i'])

    FU,FV=ecco.vector_calc.UEVNfromUXVY(FU,FV,GDS)

    latmin=-35
    latmax=90
    deltalat=1
    lonmin=-100
    lonmax=30
    deltalon=1

    lonc,latc,lone,late,FU_r=ecco.resample_to_latlon(GDS.XC,GDS.YC, FU,latmin,latmax,deltalat,lonmin,lonmax,deltalon,mapping_method='nearest_neighbor',
                                   radius_of_influence = 112000)
    lonc,latc,lone,late,FV_r=ecco.resample_to_latlon(GDS.XC,GDS.YC, FV,latmin,latmax,deltalat,lonmin,lonmax,deltalon,mapping_method='nearest_neighbor',
                                   radius_of_influence = 112000)

    
    
    

    MAG=np.sqrt(FU_r**2 + FV_r**2)[::ss,::ss]
    if vmin==None: vmin=np.min(MAG)
    if vmax==None: vmax=np.max(MAG)
    norm = Normalize(vmin=vmin,vmax=vmax)
    
    PROJ=ccrs.LambertAzimuthalEqualArea(central_latitude=0,central_longitude= -34.3)
    #### Create new axis
    POS=ax.get_position()
    FIG=ax.figure
    ax.remove()
    AX=FIG.add_axes(POS,projection=PROJ)
    
    s=AX.streamplot(lonc[::ss,::ss],latc[::ss,::ss],FU_r[::ss,::ss],FV_r[::ss,::ss],color=MAG,norm=norm,transform=ccrs.PlateCarree(),linewidth=(widthfactor*MAG/np.max(MAG)),**kwargs)

    AX.add_feature(cartopy.feature.LAND, color='k',zorder=-1)
    # AX.set_xlim(np.array(AX.get_xlim())*0.85)
    # AX.set_ylim(np.array(AX.get_ylim())*0.85)    
    # print(AX.get_xlim(),AX.get_ylim())    
    AX.set_xlim((-6464177.057381174, 6352453.523741399))
    AX.set_ylim((-4354122.528486568, 8221529.553090038))
    return s,AX

def get_ecco_forcing(variable_name,forcing_dir=None,nsteps_mean=1,calc_clim=True,show_progress=False,):
    '''
    Load raw ECCOv4r4 6-hourly surface forcing files as a numpy array, optionally calculating
    their climatology.
    
    Parameters:
    -----------
    variable_name (str): name of the surface forcing variable, from the following:
        name - documented name; description
        STANDARD ECCO
        ------------------------------------------------------
        dlw - lwdown; Downward longwave radiation in W/m^2
        dsw - swdown; Downward shortwave radiation in W/m^2
        pres - apressure; Atmospheric pressure field in N/m^2
        rain - precip; Precipitation in m/s
        spfh2m - aqh; Surface (2m) specific humidity in kg/kg
        tmp2m - atemp; Surface (2-m) air temperature in deg K
        ustr - ustress; Zonal surface wind stress in N/m^2
        vstr - vstress; Meridional surface wind stress in N/m^2
        wspeed - wspeed; Surface (10-m) wind speed in m/s
        --------------------------------------------------------
        FLUX-FORCED
        --------------------------------------------------------        
        TFLUX    - hflux   ; net upward surface heat flux (W/m2)
        oceQsw   - swflux  ; Net upward shortwave radiation (W/m2)
        oceFWflx - sflux   ; Net upward freshwater flux (m/s)
        oceSflux - saltflx ; Net upward salt flux (psu*kg/m2/s)
        oceSPflx - spflx   ; Salt tendency due to salt plume flux (g/m2/s)
        oceTAUX  - ustress ; Zonal surface wind stress (N/m^2)
        oceTAUY  - vstress ; Meridional surface wind stress (N/m^2)
        sIceLoadPatmPload_nopabar - apressure ; atmospheric pressure (N/m2)
        
        (see https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exf.html)
    
    forcing_dir (str): the location of the forcing files, most likely a directory called "input forcing"
    nsteps_mean (int, default=1): by default the forcing is at 6h frequency. If, e.g., daily averages
                                  are desired, nsteps_mean=4 instead averages over 4 6-hour steps
    calc_clim (bool, default=True): in addition to the raw forcing data, return their climatology
    show_progress (bool, default=False): provide a print update with time elapsed for every year loaded
    
    Returns:
    --------
    forc_array (np.ndarray): an (Ntot,13,90,90) array of forcing on the LLC90 grid, where Ntot is the total
                            number of outputs in the ECCO period (default 37894, determined by nsteps_mean).
    forc_array_clim (np.ndarray): an (Nyr, 13,90,90) array of the forcing climatology on the LLC90 grid, where
                            Nyr is the number of outputs in 366 days (default 1464, determined by nsteps_mean).
                            
    '''
    if forcing_dir==None:
        if variable_name in ['TFLUX','oceQsw','oceFWflx','oceSflux','oceSPflx','oceTAUX','oceTAUY','sIceLoadPatmPload_nopabar']:
            forcing_dir='./ECCOv4r4_input/other/flux-forced/forcing/'
        else:
            forcing_dir='./ECCOv4r4_input/other/forcing/'
    
    lenf=(9497*4);forc_array=np.zeros((lenf,13,90,90));forc_array_clim=np.zeros((366*4,13,90,90))
    oldi=0
    if show_progress: T0=time.time()
    nyears=len(range(1992,2018));nleaps=len([1992,1996,2000,2004,2008,2012,2016])
    for YR in range(1992,2018):
        #print(YR)
        if show_progress: print('Year :'+str(YR)+', '+str('%0.3g ' % (time.time()-T0))+' seconds elapsed')
        if YR in [1992,1996,2000,2004,2008,2012,2016]: nsteps=1464
        elif YR==2017: nsteps=1459
        else: nsteps=1460
        newi=oldi+nsteps
        if variable_name in ['TFLUX','oceQsw','oceFWflx','oceSflux','oceSPflx','oceTAUX','oceTAUY','sIceLoadPatmPload_nopabar']:
            fname=(variable_name+'_6hourlyavg_'+str(YR).zfill(4))
        else: 
            fname=('eccov4r4_'+variable_name+'_'+str(YR).zfill(4))
        forc_array[oldi:newi,:]=ecco.read_llc_to_tiles(forcing_dir,fname=fname,nl=nsteps,less_output=True)[:,0,:,:,:]
        if nsteps>=1459:
            forc_array_clim[:1459,:]=forc_array_clim[:1459,]+(forc_array[oldi:oldi+1459,:])/nyears
        if nsteps==1460:
            forc_array_clim[1459,:]=forc_array_clim[1459,]+(forc_array[oldi+1459]/(nyears-1)) # Final entry does not exist in 2017
        if nsteps==1464:
            forc_array_clim[1460:1464]=forc_array_clim[1460:1464]+(forc_array[oldi+1460:oldi+1464,:]/nleaps)            
        oldi=newi
    
    ####
    # PATCH: 2017 only has 1459 entries, which is a nightmare averaging etc. 
    # This line creates a fake final entry from the climatology as a patch, be cautious of any analysis which uses the final entry:
    forc_array[-1,:]=forc_array_clim[1459,:]
    ###
    if calc_clim:
        return forc_array.reshape(-1,nsteps_mean,13,90,90).mean(axis=1),\
          forc_array_clim.reshape(-1,nsteps_mean,13,90,90).mean(axis=1)
    else:
        return forc_array.reshape(-1,nsteps_mean,13,90,90).mean(axis=1)
    
def forcing_anom(var,var_clim,nsteps_mean=1,verbose=False):
    '''
    Take a forcing data matrix and climatology and calculate the anomaly
    '''
    lenf=(9497*4)//nsteps_mean;var_anom=np.zeros((lenf,13,90,90))
    oldi=0
    if verbose: T0=time.time()
    nyears=len(range(1992,2018));nleaps=len([1992,1996,2000,2004,2008,2012,2016])
    for YR in range(1992,2018):
        if verbose: print('Year :'+str(YR)+', '+str('%0.3g ' % (time.time()-T0))+' seconds elapsed')
        if YR in [1992,1996,2000,2004,2008,2012,2016]: nsteps=1464 #ndays=366
        #elif YR==2017: nsteps=1459 #ndays=364
        else: nsteps=1460 #ndays=365
        #newi=oldi+(ndays*4//nsteps_mean)
        newi=oldi+(nsteps//nsteps_mean)

        #var_anom[oldi:newi,:]=var[oldi:newi,:]-var_clim[:(ndays*4)//nsteps_mean,:]
        var_anom[oldi:newi,:]=var[oldi:newi,:]-var_clim[:nsteps//nsteps_mean,:]

        oldi=newi
    return var_anom    
    
def llc_facets_2d_to_compact_better(facets,extra_metadata):
    ''' This is a modified version of the function xm.utils.llc_facets_2d_to_compact
    allowing for time-varying 2D arrays such as surface forcing to be arranged 
    in a way that they can be written to the MDS file format read by MITgcm.
    
    For more info, see https://xmitgcm.readthedocs.io/en/latest/demo_writing_binary_file.html
    or the documentation for xm.utils.llc_facets_2d_to_compact
    '''
    for kfacet in range(len(facets)):
        if facets['facet' + str(kfacet)] is not None:
            if 'time' in facets['facet' + str(kfacet)].dims:
                tlen=len(facets['facet' + str(kfacet)].time)
                tmp = np.reshape(facets['facet' + str(kfacet)].values, (tlen,-1))
        if kfacet==0: flatdata=tmp.copy()
        else: flatdata = np.hstack([flatdata, tmp])
                
    flatdata=flatdata.reshape(-1)

    return flatdata

def remove_forcing_pattern(PATTERN,PATTERN_ts,forcing_component,input_dir=None,output_dir='.',return_original_forcing=False):
    '''
    Take a single spatial pattern associated with an ECCO (flux) forcing variable and write a copy of the flux forcing files with
    the pattern regressed out.
    
    Parameters:
    -----------
    PATTERN (ndarray): 1xM array consisting of the spatial pattern (e.g. an EOF). This routine only handles patterns which have been
        compressed onto a subset of the full ECCO grid (all of the wet points in the Atlantic and Arctic associated with that variable).
        To reduce a full-grid pattern (13x90x90) to this form, use the routine [idk dude you must have a routine that does this somewhere]
    PATTERN_ts (ndarray): 1xN array consisting of the time series associated with the spatial pattern (e.g. a PC). The pattern and time
        series will combine via the outer product to make a data matrix to be removed from the original data matrix
    forcing_component (str): either 'heat', 'salt', or 'wind'. 'heat' removes the pattern from the forcing variable 'TFLUX', 'salt' removes
        the pattern from the forcing variable 'oceSflux', 'wind' removes the pattern from 'oceTAUX', and 'oceTAUY' simultaneously.
        As 'wind' removes from both wind components, it must load the full forcing dataset for both 'oceTAUX' and 'oceTAUY'. This requires
        around 60GB of memory.
    output_dir (str): The directory in which to write the modified forcing files. Be sure not to overwrite the original files
    return_original_forcing (bool, optional, default: False): if True, the function returns the input forcing and output forcing for comparison
    
    Returns (conditional, return_original_forcing must be True):
    --------
    F_clim:    The (unmodified) climatology of the forcing
    F_anom:    The (modified) anomaly dataset of the forcing, with PATTERN removed
    F_anom_og: The (unmodified) anomaly dataset of the forcing
    
    '''
    T0=time.time()
    
    if   forcing_component=='heat':
        forcing_variable,idx='TFLUX',Ti
    elif forcing_component=='salt':
        forcing_variable,idx='oceFWflx',Si
    elif forcing_component=='wind':
        forcing_variable,idx='oceTAUX',np.hstack([Ui,Vi+105300])
        # Note: oceTAUY is loaded alongside and the two are concatenated
    
    print('loading forcing and calculating climatology, '+str(time.time()-T0))
    if input_dir==None:
        F_anom,F_clim=get_ecco_forcing(forcing_variable) # Note "anom" isn't actually anom, this is to save memory. Anom calculated below
    else:
        F_anom,F_clim=get_ecco_forcing(forcing_variable,forcing_dir=input_dir)
    print('getting forcing anomaly, '+str(time.time()-T0))
    F_anom  =forcing_anom(F_anom,F_clim)
        
    F_anom=F_anom.reshape(-1,13*90*90)
    
    if forcing_component=='wind':
        print('loading oceTAUY and calculating climatology')
        if input_dir==None:
            G_anom,G_clim=get_ecco_forcing('oceTAUY')
        else:
            G_anom,G_clim=get_ecco_forcing('oceTAUY',forcing_dir=input_dir)        
        
        print('getting oceTAUY anomaly')
        G_anom=forcing_anom(G_anom,G_clim)
        G_anom=G_anom.reshape(-1,13*90*90)
        G=[]; # wipe G to save memory
        
        #F_clim=np.hstack([F_clim,G_clim])
        F_anom=np.hstack([F_anom,G_anom])
    
    if return_original_forcing:
        F_anom_og=F_anom.copy()
 
    print('removing pattern, '+str(time.time()-T0))

    F_anom[:,idx]=F_anom[:,idx]-PATTERN_ts[:,None].dot(PATTERN[None,:])

    # Now just need to write it and we're off to the races
    oldi=0
    extra_metadata=xm.utils.get_extra_metadata(domain='llc',nx=90)
    print('writing output, '+str(time.time()-T0))
    for YR in range(1992,2018):
        if YR in [1992,1996,2000,2004,2008,2012,2016]: nsteps=1464;
        elif YR==2017: nsteps=1459;
        else: nsteps=1460;
        
        newi=oldi+nsteps
        if forcing_component=='wind':
            # For wind, have to write a single data matrix to a pair of forcing files
            FOR_YRU=F_clim[:nsteps,:]+(F_anom[oldi:newi,:105300]).reshape(-1,13,90,90)
            FOR_YRU=xr.DataArray(FOR_YRU,dims=['time','face','j','i'])            
            FOR_YRV=G_clim[:nsteps,:]+(F_anom[oldi:newi,105300:]).reshape(-1,13,90,90)
            FOR_YRV=xr.DataArray(FOR_YRV,dims=['time','face','j','i'])            
            facetsU=xm.utils.rebuild_llc_facets(FOR_YRU,extra_metadata)
            facetsV=xm.utils.rebuild_llc_facets(FOR_YRV,extra_metadata)
            compactU=llc_facets_2d_to_compact_better(facetsU,extra_metadata=extra_metadata)
            compactV=llc_facets_2d_to_compact_better(facetsV,extra_metadata=extra_metadata)
            
            print('writing file '+output_dir+'oceTAUX_6hourlyavg_'+str(YR).zfill(4)+', '+str(time.time()-T0))
            xm.utils.write_to_binary(compactU,output_dir+'oceTAUX_6hourlyavg_'+str(YR).zfill(4))
            print('writing file '+output_dir+'oceTAUY_6hourlyavg_'+str(YR).zfill(4)+', '+str(time.time()-T0))
            xm.utils.write_to_binary(compactV,output_dir+'oceTAUY_6hourlyavg_'+str(YR).zfill(4))
        else:  
            FOR_YR=F_clim[:nsteps,:]+(F_anom[oldi:newi,:].reshape(-1,13,90,90))
            FOR_YR=xr.DataArray(FOR_YR,dims=['time','face','j','i'])
            facets1=xm.utils.rebuild_llc_facets(FOR_YR,extra_metadata)
            compact1=llc_facets_2d_to_compact_better(facets1,extra_metadata=extra_metadata)
            print('writing file '+output_dir+forcing_variable+'_6hourlyavg_'+str(YR).zfill(4)+', '+str(time.time()-T0))
            xm.utils.write_to_binary(compact1,output_dir+forcing_variable+'_6hourlyavg_'+str(YR).zfill(4))

        oldi=newi
    print('done! '+str(time.time()-T0))
    if return_original_forcing:
        return F_clim,F_anom_og,F_anom