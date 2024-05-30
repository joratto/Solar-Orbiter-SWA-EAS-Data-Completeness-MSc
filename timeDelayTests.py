import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def lag_finder(y1, y2, sr):
    # Courtesy of Reveille on stackoverflow
    n = len(y1)

    corr = sp.signal.correlate(y2, y1, mode='same') / np.sqrt(sp.signal.correlate(y1, y1, mode='same')[int(n/2)] * sp.signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()

#lag_finder(y1,y2,sr)

# Sine sample with some noise and copy to y1 and y2 with a 1-second lag
'''
sr = 1024
y = np.linspace(0, 2*np.pi, sr)
y = np.tile(np.sin(y), 5)
y += np.random.normal(0, 5, y.shape)
y1 = y[sr:4*sr]
y2 = y[:3*sr]

plt.plot(y1)
plt.plot(y2)

lag_finder(y1, y2, sr)
'''

def delayFinder(f,g,sr): 
    # f and g are the functions to cross-correlate
    # sr is the samplerate
    # positive delays when f is ahead
    corr = sp.signal.correlate(f,g,mode='full') # the correlation distribution
    print('correlation={}'.format(corr))
    lags = sp.signal.correlation_lags(f.size,g.size,mode='full')/sr # range of possible lags, scaled by samplerate
    print('lags={}'.format(lags))
    print(lags.size)
    delay = lags[np.argmax(corr)]
    print('delay={}s'.format(delay))
    return delay, lags, corr

def Gauss(x,a=1,b=0,c=1):
    return a*np.exp(-(x-b)**2/(2*c**2))
    # a	=	height of the curve's peak
    # b	=	the position of the center of the peak
    # c	=	the standard deviation

N = 100
T = 610 # s
sr = N/T
x = np.linspace(0,T,N)
dt = 61.5926535897932384626433 # s
a = 1
b = 300
c = 50
f = Gauss(x,a,b,c)
g = Gauss(x+dt,a*0.8,b,c*0.8)

#print(delayFinder(f,g))

Nplots = 2
fig, axs = plt.subplots(Nplots)

axs[0].plot(x,f)
axs[0].plot(x,g)
axs[0].legend(["f", "g"])

delay, lags, corr = delayFinder(f,g,sr)

axs[1].plot(lags,corr)

plt.show()