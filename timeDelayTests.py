import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import random
import seaborn as sns

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
mode = 'full'

def ccf(f,g,sr):
    # f and g are the functions to cross-correlate
    # sr is the samplerate
    # positive delays when f is ahead
    corr = sp.signal.correlate(f,g,mode=mode) # the correlation distribution
    lags = sp.signal.correlation_lags(f.size,g.size,mode=mode)/sr # range of possible lags, scaled by samplerate

    return lags, corr

def delayFinder(lags, corr): 
    # lags = array of possible delays (tau) given by ccf()
    # corr = ccf given by ccf()
    # print('delay={}s'.format(delay))
    delay_index = np.argmax(corr)
    delay = lags[delay_index]
    corrmax = corr[delay_index]

    return delay, corrmax

def Gauss(x,a=1,b=0,c=1):
    return a*np.exp(-(x-b)**2/(2*c**2))
    # a	=	height of the curve's peak
    # b	=	the position of the center of the peak
    # c	=	the standard deviation

N = 3200
T = 400 # s
sr = N/T
x = np.linspace(-T/2,T/2,N)
dt = -100 # s
a = 1
b = 0
c = 25
f = Gauss(x,a,b,c)
g = Gauss(x-dt,a*0.75,b,c*0.5)

noiseamp = 0.00
f += np.random.normal(0, noiseamp, f.shape)
g += np.random.normal(0, noiseamp, g.shape)

#print(delayFinder(f,g))

def bootstrapping(f,g,meanStep=3,N=100,sr=8):
    Nf = len(f)
    Ng = len(g)
    
    if Nf != Ng:
        raise ValueError('Nf != Ng')
    
    meanStep = int(meanStep)
    minStep = 1
    maxStep = 2*meanStep - 1

    tauArray = []
    for i in range(N):
        fi = []
        gi = []
        j = 0
        
        randomStep = random.randint(minStep,maxStep)
        while j + randomStep < Nf:
            fi.append(f[j + randomStep])
            gi.append(g[j + randomStep])
            j += randomStep
            randomStep = random.randint(minStep,maxStep)
        
        print('\n{},{}'.format(len(fi),len(gi)))
        lags, corr = ccf(np.array(fi),np.array(gi),sr/meanStep)
        delay, corrmax = delayFinder(lags, corr)
        tauArray.append(delay)

    return tauArray

noiseamp = 1

# f += np.random.normal(0, noiseamp, f.shape)
# g += np.random.normal(0, noiseamp, g.shape)

N = 1000
# tauArray = noiseTesting(noiseamp, N=N, sr=sr)
tauArray = bootstrapping(f, g, meanStep=3, N=N, sr=sr)
print(tauArray)

mu = np.mean(tauArray)
sig = np.std(tauArray)
print(mu)
print(sig)


Nplots = 2
fig, axs = plt.subplots(Nplots, figsize=(6,4))
sns.set_theme(style='ticks')
fig.tight_layout()

linecolor = 'gray'
textpad = 0.05
textbbox = dict(facecolor='white', alpha=0.5, edgecolor='gray')
textbbox = dict(facecolor='white', alpha=0, edgecolor='gray')
fontsize = 12

axs[0].set_title(r'$\tau_0$'+' = {}s'.format(-dt), fontsize=fontsize)

ax = axs[0]
ax.plot(x,f)
ax.plot(x,g)
ax.legend([r'$f(t)$', r'$g(t)$'])
ax.grid()

# trans = ax.get_xaxis_transform()
ax.vlines(b,0,Gauss(b,a,b,c),linestyle='--',color=linecolor)
ax.vlines(b+dt,0,Gauss(b,a*0.75,b,c*0.5),linestyle='--',color=linecolor)
# plt.text(b, 1.1, 'hello', transform=trans)
# plt.text(b+dt, 1.1, 'world', transform=trans)
ax.text(b, Gauss(b,a,b,c)+textpad, r'$t$'+' = {}s'.format(b), ha='center', bbox=textbbox, fontsize=fontsize)
ax.text(b+dt, Gauss(b,a*0.75,b,c*0.5)+textpad, r'$t=-\tau_0$'+ ' = {}s'.format(b+dt), ha='center', bbox=textbbox, fontsize=fontsize)
ax.set_ylim(-0.2,1.2)
ax.set_ylabel('Normalised Signal', fontsize=fontsize)

lags, corr = ccf(f,g,sr)
delay, corrmax = delayFinder(lags, corr)

print(delay)
print(corrmax)

ax.set_xlabel('Time '+r'$t$'+' (seconds)', fontsize=fontsize)

ax = axs[1]
ax.plot(lags,corr/corrmax,color='r')
ax.legend(['CCF'+r'$(\tau)$'])
ax.grid()

# trans = ax.get_xaxis_transform()
ax.vlines(delay,0,1,linestyle='--',color=linecolor)
# plt.text(delay, 1.1, 'bye bye', transform=trans)5
ax.text(delay, 1+textpad, r'$\tau_r$'+' = {:.1f}s'.format(delay), ha='center', bbox=textbbox, fontsize=fontsize)
ax.set_ylim(-0.2,1.2)
ax.set_ylabel('Normalised CCF', fontsize=fontsize)

ax.errorbar(mu,1,yerr=0,xerr=sig)

ax.set_xlabel('Time Delay '+r'$\tau$'+' (seconds)', fontsize=fontsize)

plt.show()


def noiseTesting(noiseamp,N,sr):
    tauArray = []
    for i in range(N):
        f = Gauss(x,a,b,c)
        g = Gauss(x-dt,a*0.75,b,c*0.5)

        f += np.random.normal(0, noiseamp, f.shape)
        g += np.random.normal(0, noiseamp, g.shape)

        lags, corr = ccf(f,g,sr)
        delay, corrmax = delayFinder(lags, corr)

        tauArray.append(delay)
    
    return tauArray


plt.hist(tauArray,bins=10,density=True)
N = 1000
x = np.linspace(min(tauArray),min(tauArray),N)
plt.plot(x,Gauss(x,1,mu,sig))
plt.xlabel('Time Delay '+r'$\tau$'+ ' (seconds)', fontsize=fontsize)
plt.ylabel('Normalised Probability Density', fontsize=fontsize)
plt.grid()

plt.show()

