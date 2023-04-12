from matplotlib import rcParams
rcParams['figure.figsize']=(8,8)
rcParams['font.family']='STIXGeneral'
rcParams['font.size']=25
rcParams['mathtext.fontset']='stix'
rcParams['legend.numpoints']=1
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import PixCoord, PolygonSkyRegion, PolygonPixelRegion
from regions import CircleSkyRegion, PixCoord, CirclePixelRegion
from astropy.convolution import convolve
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from regions import Regions

#path to data, galaxy to check background on
path1 = 'v0p7p3/background_regions/'
path = 'v0p7p3/anchored/'
gal = 'ngc1566'

#open selected data
with fits.open(path+gal+'_F770W_anchored.fits') as hdul:
    f770 = hdul['SCI'].data
    h770 = hdul['SCI'].header
wcs770 = WCS(h770).celestial
with fits.open(path+gal+'_F1130W_anchored.fits') as hdul:
    f1130 = hdul['SCI'].data
    h1130 = hdul['SCI'].header
wcs1130 = WCS(h1130).celestial
with fits.open(path+gal+'_F2100W_anchored.fits') as hdul:
    f2100 = hdul['SCI'].data
    h2100 = hdul['SCI'].header
wcs2100 = WCS(h2100).celestial

#load background region
back_reg = Regions.read(path1+gal+'_background.reg', format='ds9')

pix_reg = back_reg[0].to_pixel(wcs770)
print(pix_reg.center.x)
x0 = pix_reg.center.x
y0 = pix_reg.center.y
width = pix_reg.width
height = pix_reg.height

#show background regions
cutout770 = Cutout2D(f770, (x0,y0), (width,height))
cutout1130 = Cutout2D(f1130, (x0,y0), (width,height))
cutout2100 = Cutout2D(f2100, (x0,y0), (width,height))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,10))
ax1.imshow(cutout770.data)
ax1.set_title('MIRI F770W')
ax2.imshow(cutout1130.data)
ax2.set_title('MIRI F1130W')
ax3.imshow(cutout2100.data)
ax3.set_title('MIRI F2100W')
plt.show()

#Plot histograms
b770 = cutout770.data.ravel()
b1130 = cutout1130.data.ravel()
b2100 = cutout2100.data.ravel()

# Empirical average and variance are computed
a770 = np.mean(b770)
v770 = np.var(b770)
std770 = np.std(b770)
# From that, we know the shape of the fitted Gaussian.
pdf_x = np.linspace(-2,2,100)
pdf_y_770 = 1.0/np.sqrt(2*np.pi*v770)*np.exp(-0.5*(pdf_x-a770)**2/v770)

a1130 = np.mean(b1130)
v1130 = np.var(b1130)
std1130 = np.std(b1130)
pdf_y_1130 = 1.0/np.sqrt(2*np.pi*v1130)*np.exp(-0.5*(pdf_x-a1130)**2/v1130)

a2100 = np.mean(b2100)
v2100 = np.var(b2100)
std2100 = np.std(b2100)
pdf_y_2100 = 1.0/np.sqrt(2*np.pi*v2100)*np.exp(-0.5*(pdf_x-a2100)**2/v2100)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30,10))
ax1.hist(b770, bins=100, density=True)
ax1.plot(pdf_x, pdf_y_770)
ax1.set_xlim(-1, 1)
ax1.text(-0.75, 2, 'av={:.4f}'.format(a770))
ax1.text(-0.75, 1.7, 'var={:.4f}'.format(v770))
ax1.text(-0.75, 1.5, 'var={:.4f}'.format(std770))
ax1.set_title('MIRI F770W')
ax2.hist(b1130, bins=100, density=True)
ax2.plot(pdf_x, pdf_y_1130)
ax2.set_xlim(-1, 1)
ax2.text(-0.75, 1.5, 'av={:.4f}'.format(a1130))
ax2.text(-0.75, 1.3, 'var={:.4f}'.format(v1130))
ax2.text(-0.75, 1.1, 'std={:.4f}'.format(std1130))
ax2.set_xlim(-1, 1)
ax2.set_title('MIRI F1130W')
ax3.hist(b2100, bins=100, density=True)
ax3.plot(pdf_x, pdf_y_2100)
ax3.set_xlim(-1, 1)
ax3.text(-0.75, .9, 'av={:.4f}'.format(a2100))
ax3.text(-0.75, .7, 'var={:.4f}'.format(v2100))
ax3.text(-0.75, .5, 'std={:.4f}'.format(std2100))
ax3.set_xlim(-1, 1)
ax3.set_title('MIRI F2100W')
fig.savefig(path1+gal+'background.jpeg')
