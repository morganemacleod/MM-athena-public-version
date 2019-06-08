import numpy as np
from astropy.io import ascii
import athena_read as ar

def read_trackfile(fn,m1=0,m2=0):
    orb=ascii.read(fn)
    print "reading orbit file for planet wind simulation..."
    if m1==0:
        m1 = orb['m1']
    if m2==0:
        m2 = orb['m2']

    orb['sep'] = np.sqrt(orb['x']**2 + orb['y']**2 + orb['z']**2)

    orb['r'] = np.array([orb['x'],orb['y'],orb['z']]).T
    orb['rhat'] = np.array([orb['x']/orb['sep'],orb['y']/orb['sep'],orb['z']/orb['sep']]).T

    orb['v'] = np.array([orb['vx'],orb['vy'],orb['vz']]).T
    orb['vmag'] = np.linalg.norm(orb['v'],axis=1)
    orb['vhat'] = np.array([orb['vx']/orb['vmag'],orb['vy']/orb['vmag'],orb['vz']/orb['vmag']]).T

    orb['xcom'] = m2*orb['x']/(m1+m2)
    orb['ycom'] = m2*orb['y']/(m1+m2)
    orb['zcom'] = m2*orb['z']/(m1+m2)
    
    orb['vxcom'] = m2*orb['vx']/(m1+m2)
    orb['vycom'] = m2*orb['vy']/(m1+m2)
    orb['vzcom'] = m2*orb['vz']/(m1+m2)
    
    orb['rcom'] = np.array([orb['xcom'],orb['ycom'],orb['zcom']]).T
    orb['vcom'] = np.array([orb['vxcom'],orb['vycom'],orb['vzcom']]).T
    
    return orb


def read_data(fn,orb,
              m1=0,m2=0,rsoft2=0.1,level=0,
              get_cartesian=True,get_cartesian_vel=True,
             x1_min=None,x1_max=None,
             x2_min=None,x2_max=None,
             x3_min=None,x3_max=None,
             gamma=5./3.):
    """ Read spherical data and reconstruct cartesian mesh for analysis/plotting """
    
    print "read_data...reading file",fn
    
    
    d = ar.athdf(fn,level=level,subsample=True,
                 x1_min=x1_min,x1_max=x1_max,
                 x2_min=x2_min,x2_max=x2_max,
                 x3_min=x3_min,x3_max=x3_max) # approximate arrays by subsampling if level < max
    print " ...file read, constructing arrays"
    print " ...gamma=",gamma
    
    # current time
    t = d['Time']
    # get properties of orbit
    rcom,vcom = rcom_vcom(orb,t)

    if m1==0:
        m1 = np.interp(t,orb['time'],orb['m1'])
    if m2==0:
        m2 = np.interp(t,orb['time'],orb['m2'])

    data_shape = (len(d['x3v']),len(d['x2v']),len(d['x1v']))
   
       
    # MAKE grid based coordinates
    d['gx1v'] = np.zeros(data_shape)
    for i in range(data_shape[2]):
        d['gx1v'][:,:,i] = d['x1v'][i]
    
    d['gx2v'] = np.zeros(data_shape)
    for j in range(data_shape[1]):
        d['gx2v'][:,j,:] = d['x2v'][j]

    d['gx3v'] = np.zeros(data_shape)
    for k in range(data_shape[0]):
        d['gx3v'][k,:,:] = d['x3v'][k]
    
    
    ####
    # GET THE VOLUME 
    ####
    
    ## dr, dth, dph
    d1 = d['x1f'][1:] - d['x1f'][:-1]
    d2 = d['x2f'][1:] - d['x2f'][:-1]
    d3 = d['x3f'][1:] - d['x3f'][:-1]
    
    # grid based versions
    gd1 = np.zeros(data_shape)
    for i in range(data_shape[2]):
        gd1[:,:,i] = d1[i]
    
    gd2 = np.zeros(data_shape)
    for j in range(data_shape[1]):
        gd2[:,j,:] = d2[j]

    gd3 = np.zeros(data_shape)
    for k in range(data_shape[0]):
        gd3[k,:,:] = d3[k]
    
    # AREA / VOLUME 
    sin_th = np.sin(d['gx2v'])
    d['dA'] = d['gx1v']**2 * sin_th * gd2*gd3
    d['dvol'] = d['dA'] * gd1
    
    # free up d1,d2,d3
    del d1,d2,d3
    del gd1,gd2,gd3
    
    
    ### 
    # CARTESIAN VALUES
    ###
    if(get_cartesian or get_torque or get_energy):
        print "...getting cartesian arrays..."
        # angles
        cos_th = np.cos(d['gx2v'])
        sin_ph = np.sin(d['gx3v'])
        cos_ph = np.cos(d['gx3v']) 
        
        # cartesian coordinates
        d['x'] = d['gx1v'] * sin_th * cos_ph 
        d['y'] = d['gx1v'] * sin_th * sin_ph 
        d['z'] = d['gx1v'] * cos_th

        if(get_cartesian_vel or get_torque or get_energy):
            # cartesian velocities
            d['vx'] = sin_th*cos_ph*d['vel1'] + cos_th*cos_ph*d['vel2'] - sin_ph*d['vel3'] 
            d['vy'] = sin_th*sin_ph*d['vel1'] + cos_th*sin_ph*d['vel2'] + cos_ph*d['vel3'] 
            d['vz'] = cos_th*d['vel1'] - sin_th*d['vel2']  
            
        del cos_th, sin_th, cos_ph, sin_ph
    
    return d



def get_midplane_theta(myfile,level=0):
    dblank=ar.athdf(myfile,level=level,quantities=[],subsample=True)

    # get closest to midplane value
    return dblank['x2v'][ np.argmin(np.abs(dblank['x2v']-np.pi/2.) ) ]


def get_plot_array_midplane(arr):
    return np.append(arr,[arr[0]],axis=0)


def rcom_vcom(orb,t):
    """pass a pm_trackfile.dat that has been read, time t"""
    rcom =  np.array([np.interp(t,orb['time'],orb['rcom'][:,0]),
                  np.interp(t,orb['time'],orb['rcom'][:,1]),
                  np.interp(t,orb['time'],orb['rcom'][:,2])])
    vcom =  np.array([np.interp(t,orb['time'],orb['vcom'][:,0]),
                  np.interp(t,orb['time'],orb['vcom'][:,1]),
                  np.interp(t,orb['time'],orb['vcom'][:,2])])
    
    return rcom,vcom

def pos_secondary(orb,t):
    x2 = np.interp(t,orb['time'],orb['x'])
    y2 = np.interp(t,orb['time'],orb['y'])
    z2 = np.interp(t,orb['time'],orb['z'])
    return x2,y2,z2