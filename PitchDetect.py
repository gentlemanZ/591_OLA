"""
 File: cs591Utilities.py
 Author: Wayne Snyder

 Date: 1/28/17
 Purpose: This collects together the most important algorithms used in
          CS 591, in order to work interactively; for the most part
          signals are manipulated as arrays, not as wave files.
          This file assumes you have scipy and numpy.
          
          The main difference from previous version is that
          we are using numpy arrays exclusively. 
"""

import array
import contextlib
import wave
import numpy as np
import matplotlib.pyplot as plt
#import PhaseVocoder
from scipy.io import wavfile
import pygame
import argparse
import sys
import warnings
from pygame.locals import *

from numpy import pi, sin, cos, exp, abs
from scipy.io.wavfile import read, write


"""
 Basic parameters
"""


numChannels   = 1                      # mono
sampleWidth   = 2                      # in bytes, a 16-bit short
SR            = 44100                  #  sample rate
MAX_AMP       = (2**(8*sampleWidth - 1) - 1)    #maximum amplitude is 2**15 - 1  = 32767
MIN_AMP       = -(2**(8*sampleWidth - 1))       #min amp is -2**15


"""
 Basic utilities
"""

# round to 4 decimal places

def round4(x):
    return round(x+0.00000000001,4)
    
def clipZero(x):
    return (x if(x >= 0) else 0)
    
    
def clip(x):
    if(x > MAX_AMP):
        return MAX_AMP
    elif(x < MIN_AMP):
        return MIN_AMP
    else:
        return int(x)
        
"""
 File I/O        
"""

# Read a wave file and return the entire file as a standard array
# infile = filename of input Wave file
# If you set withParams to True, it will return the parameters of the input file

def readWaveFile(infile,withParams=False,asNumpy=True):
    with contextlib.closing(wave.open(infile)) as f:
        params = f.getparams()
        frames = f.readframes(params[3])
        if(params[0] != 1):
            print("Warning in reading file: must be a mono file!")
        if(params[1] != 2):
            print("Warning in reading file: must be 16-bit sample type!")
        if(params[2] != 44100):
            print("Warning in reading file: must be 44100 sample rate!")
    if asNumpy:
        X = array.array('h', frames)
        X = np.array(X,dtype='int16')
    else:  
        X = array.array('h', frames)
    if withParams:
        return X,params
    else:
        return X


#  Symmetric to last one, write an array of ints out to a file named
#  fname; if values exceed 16 bit int range, will be clipped.
        
def writeWaveFile(fname, X):
    X = [clip(x) for x in X]
    params = [1,2, SR , len(X), "NONE", None]
    data = array.array("h",X)
    with contextlib.closing(wave.open(fname, "w")) as f:
        f.setparams(params)
        f.writeframes(data.tobytes())
    print(fname + " written.")
    


def makeSignal(spectrum, duration):
    X = [0]*int(SR*duration)  
    for (f,A,phi) in spectrum:
        for i in range(len(X)):           
            X[i] += MAX_AMP * A * np.sin( 2 * np.pi * f * i / SR + phi)
    return X


"""    
 Display a signal with various options
   
   X is an array of samples
   xUnits are scale of x axis: "Seconds" (default), "Milliseconds", or "Samples"
   yUnits are "Relative" [-1..1] (default) or "Absolute" [-MAX_AMP-1 .. MAX_AMP])
   left and right delimit range of signal displayed: [left .. right) in xUnits
   width is width of figure (height is 3)


"""

def displaySignal(X, left = 0, right = -1, title='Signal Window for X',xUnits = "Seconds", yUnits = "Relative",width=10):

        
    minAmplitude = -(2**15 + 100)        # just to improve visibility of curve
    maxAmplitude = 2**15 + 300    
    
    if(xUnits == "Samples"):
        if(right == -1):
            right = len(X)
        T = range(left,right)
        Y = X[left:right]
    elif(xUnits == "Seconds"):
        if(right == -1):
            right = len(X)/44100
        T = np.arange(left, right, 1/44100)
        leftSampleNum = int(left*44100)
        Y = X[leftSampleNum:(leftSampleNum + len(T))]
    elif(xUnits == "Milliseconds"):
        if(right == -1):
            right = len(X)/44.1
        T = np.arange(left, right, 1/44.1)
        leftSampleNum = int(left*44.1)
        Y = X[leftSampleNum:(leftSampleNum + len(T))]
    else:
        print("Illegal value for xUnits")
        
    if(yUnits == "Relative"):
        minAmplitude = -1.003            # just to improve visibility of curve
        maxAmplitude = 1.01
        Y = [x/32767 for x in Y]

    fig = plt.figure(figsize=(width,4))   # Set x and y dimensions of window: may need to redo for your display
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = plt.axes()
    ax.set_xlabel(xUnits)
    ax.set_ylabel(yUnits + ' Amplitude')
    ax.set_ylim([minAmplitude,maxAmplitude])
    ax.set_xlim([left, right])
    plt.axhline(0, color='black')      # draw the 0 line in black
    plt.plot(T,Y) 
    if(    (xUnits == "Samples" and (right - left) < 51)
        or (xUnits == "Seconds" and (right - left) < 0.001)
        or (xUnits == "Milliseconds" and (right - left) < 1) ):
            plt.plot(T,Y, 'bo')                     
    plt.grid(True)                     # if you want dotted grid lines
    plt.show()

    

def makeSpectrum(instr,freq=220):
    if(instr=="triangle"):
        return ([(freq,1.0,0.0),     # triples will be converted arrays
        (freq*3,-1/(9),0.0), 
        (freq*5,1/(25),0.0), 
        (freq*7,-1/(49),0.0), 
        (freq*9,1/(81),0.0), 
        (freq*11,-1/(121),0.0), 
        (freq*13,1/(13*13),0.0)])
    elif(instr=="square"):
        return ([(freq,2/(np.pi),0.0), 
        (freq*3,2/(3*np.pi),0.0), 
        (freq*5,2/(5*np.pi),0.0), 
        (freq*7,2/(7*np.pi),0.0), 
        (freq*9,2/(9*np.pi),0.0), 
        (freq*11,2/(11*np.pi),0.0), 
        (freq*13,2/(13*np.pi),0.0),
        (freq*15,2/(15*np.pi),0.0),
        (freq*17,2/(17*np.pi),0.0),
        (freq*19,2/(19*np.pi),0.0),
        (freq*21,2/(21*np.pi),0.0)])
    elif(instr=="clarinet"):
        return ([(freq,0.314,0.0), 
        (freq*3,.236,0.0), 
        (freq*5,0.157,0.0), 
        (freq*7,0.044,0.0), 
        (freq*9,0.157,0.0), 
        (freq*11,0.038,0.0), 
        (freq*13,0.053,0.0)] ) 
    elif(instr=="bell"):
        return ([(freq,0.1666,0.0), 
        (freq*2,0.1666,0.0), 
        (freq*3,0.1666,0.0), 
        (freq*4.2,0.1666,0.0), 
        (freq*5.4,0.1666,0.0), 
        (freq*6.8,0.1666,0.0)])  
    elif(instr=="steelstring"):
        return ([(freq*0.7272, .00278,0.0),
                (freq, .0598,0.0),
                (freq*2, .2554,0.0),
                (freq*3, .0685,0.0),
                (freq*4, .0029,0.0),
                (freq*5, .0126,0.0),
                (freq*6, .0154,0.0),
                (freq*7, .0066,0.0),
                (freq*8, .0033,0.0),
                (freq*11.0455, .0029,0.0),
                (freq*12.0455, .0094,0.0),
                (freq*13.0455, .0010,0.0),
                (freq*14.0455, .0106,0.0),
                (freq*15.0455, .0038,0.0)])
    else:
        return ([])   
    

  
# wrapper around numpy fft to produce real spectrum
# This will produce

def realFFT(X):
    return 2*abs(np.fft.rfft(X))/len(X)
       
# return the phase spectrum

def phaseFFT(X):
    return [np.angle(x) for x in np.fft.rfft(X)]
    
# return fft coefficients in polar form

def polarFFT(X):
    return [np.polar(2*x/len(X)) for x in np.fft.rfft(X)]
 
# This takes a list of frequencies F (for the x axis) and a list of 
#   corresponding amplitudes S
 
def displaySpectrum(F,S,logscaleX = False, logscaleY = False, printSpectrum = False):    
    fig = plt.figure(figsize=(10,3))          # Set x and y dimensions of window: may need to redo for your display
    fig.suptitle('Spectrum', fontsize=14, fontweight='bold')
    ax = plt.axes()
    if (max(S) > 2):
        S = [s/32767 for s in S]
    if(logscaleX):
        ax.set_xscale('log')
    if(logscaleY):
        ax.set_yscale('log')
    rangeF = max(F) - min(F)
    rangeS = max(S) - min(S)
    if(logscaleX):
        ax.set_xlim([1,max(F)+(rangeF/10.0)])
    else:
        ax.set_xlim([0,max(F)+(rangeF/10.0)])
    ax.set_ylim([0,max(S)+(rangeS/10.0)])
    if(logscaleX):
        ax.set_xlabel('Frequency (Log Scale)')
    else:
        ax.set_xlabel('Frequency')
    if(logscaleY):
        ax.set_ylabel('Amplitude (Log Scale)')
    else:
        ax.set_ylabel('Amplitude')

    plt.plot(F,S)
    plt.show()
    if(printSpectrum):
        print("\nFreq\tAmp\n")
        for f in range(len(S)):
            if(abs(S[f]) > 0.01):
                print(str(F[f]) + "\t" + str(round4(S[f])))

# Display a spectrum when there are relatively few frequencies
# this will work with separate lists of frequencies F and amplitudes S,
# or as pairs (f,A) or triples (f,A,phi).  When using tuples, 
# put as argument F. This will work for negative amplitudes or 
# negative frequencies, and for relative or absolute frequencies.

# Examples:

#  displayLollipopSpectrum([2,3,6],[0.6,0.3,0.1])

#  displayLollipopSpectrum([(2,0.6),(3,0.3),(6,0.1)])

#  displayLollipopSpectrum([(2,0.6,0.0),(3,0.3,3.14),(6,0.1,0.0)])

#  displayLollipopSpectrum(makeSpectrum('steelstring',2200))   
   
def displayLollipopSpectrum(F,S=[],logscaleX = False, logscaleY = False):
    fig = plt.figure(figsize=(10,3))          # Set x and y dimensions of window: may need to redo for your display
    fig.suptitle('Spectrum', fontsize=14, fontweight='bold')
    ax = plt.axes()
    # convert from pairs or triples to F and S

    if(type(F[0])==tuple or type(F[0])==list):
        if(len(F[0]) == 3):
            S = [a for (f,a,phi) in F]
            F= [f for (f,a,phi) in F]
        elif(len(F[0]) == 2):
            S = [a for (f,a) in F]
            F= [f for (f,a) in F]

    if(logscaleX):
        ax.set_xscale('log')
        minX = 1
        maxX = 22050
    else:
        if(max(F) < 0):
            maxX = 0.0
        else:
            maxX = min(SR/2,max(F) * 1.2)
        if(min(F) < 0):          # negative frequencies
            minX = min(F) * 1.2
        else:
            minX = 0

    if(logscaleY):
        ax.set_yscale('log')
        minY = 1
        maxY = 32767
    else:
        if(min(S) < 0):          # negative amplitudes
            minY = min(S) * 1.2
            plt.plot([minX,maxX],[0,0],color='k', linestyle='-', linewidth=1)
        else:
            minY = 0
        if(max(S) < 0):
            maxY = 0.0
        else:
            maxY = max(S) * 1.2
        
    ax.set_xlim([minX,maxX])
    ax.set_ylim([minY,maxY])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')

    for i in range(len(F)):
        if(S[i] >= 0.001):
            plt.plot([F[i], F[i]], [0,S[i]], color='k', linestyle='-', linewidth=1)
            plt.plot([F[i]], [S[i]],'ro')
        elif(S[i] <= -0.001):
            plt.plot([F[i], F[i]], [S[i],0], color='k', linestyle='-', linewidth=1)
            plt.plot([F[i]], [S[i]],'ro')
    plt.show()
     
# display the spectrum of the signal window X
# The frequency bins will be for frequencies 0, w, 2w, ...., up to Nyquist Limit
# where w = 44100/len(X) = frequency of a sine wave whose period = length of X.

def displaySignalSpectrum(X,limit=22050,logscaleX=False):
    S = realFFT(X)
    incr = SR/len(X)
    F = [i*incr for i in range(len(S))]
    L = int(limit//incr)
    displaySpectrum(F[:L], S[:L],logscaleX)
    
def displaySignalLollipopSpectrum(X,limit=22050,logscaleX=False):
    S = realFFT(X)
    S = [s/32767 for s in S]
    incr = SR/len(X)
    F = [i*incr for i in range(len(S))]
    L = int(limit//incr)
    displayLollipopSpectrum(F[:L], S[:L],logscaleX)
    

#hw06 Problem 1:
    
def acorr(X, lag):
    sum = 0
    for i in range(len(X) - lag):
       sum += X[i]*X[i+lag]
    return sum/(len(X)-lag)

def pitchDetector(X):
    #normalize signal
    normaled_X = [x/32767 for x in X]
    N = len(X)//2
    A = [0]*N
    for k in range(N):
        A[k] = acorr(normaled_X,k)
    Standered = A[0]
    for i in range(N):
        A[i] = A[i]/Standered
    plt.plot(A)       # should draw a blue curve
    
    plt.axhline(0, color='black')      # draw the 0 line in black

    # now find the peaks and store the lags in L and the coefficients in C
    #print(("peaks"),end='                                ')
    print('Frequency')
    for i in range(1,len(A)-1):
        if A[i]>A[i-1] and A[i]> A[i+1]:
            L = i
            C = A[i]
            plt.plot(L,C, 'ro')# will draw red dots exactly at the peaks
            #print((i,A[i]),end='          ')
            print(SR/i)

    # now print out the peaks and the corresponding frequencies
 
#hw06 Problem 2:
    
def CalcParabolaVertex(x1, y1,x2,y2,x3,y3):
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    xv = -B / (2*A);
    yv = C - B*B / (4*A);
    return(xv,yv)


def pitchDetector2(X):
    #normalize signal
    normaled_X = [x/32767 for x in X]
    N = len(X)//2
    A = [0]*N
    for k in range(N):
        A[k] = acorr(normaled_X,k)
    Standered = A[0]
    for i in range(N):
        A[i] = A[i]/Standered
    plt.plot(A)       # should draw a blue curve
    
    plt.axhline(0, color='black')      # draw the 0 line in black

    # now find the peaks and store the lags in L and the coefficients in C
    #print(("peaks"),end='                                 ')
    print('Frequency')
    preL = 0
    for i in range(1,len(A)-1):
        if A[i]>A[i-1] and A[i]> A[i+1]:
            if A[i] >0.9:
                x1 = i-1
                y1 = A[i-1]
                x2 = i
                y2 = A[i]
                x3 = i+1
                y3 = A[i+1]
                (xv,yv) = CalcParabolaVertex(x1, y1,x2,y2,x3,y3)
                L = i
                C = A[i]
                Segment_length = (i - preL)*2
                preL = L
                plt.plot(L,C, 'ro')# will draw red dots exactly at the peaks
                #print((i,A[i]),end='          ')
                print(SR/i)
                break
            
            
#Final project(pitch detecting):
def segmentStretching(X,Segment_length):
    #normalize signal
    normaled_X = [x/32767 for x in X]
    N = len(X)
    A = [0]*N
    for k in range(N):
        A[k] = acorr(normaled_X,k)
    Standered = A[0]
    for i in range(N):
        A[i] = A[i]/Standered
    
    preL = 0
    result =[]
    segment =[]
    for i in range(1,len(A)-1):
        if A[i]>A[i-1] and A[i]> A[i+1]:
            if A[i] >0.9:
                L = i
                #C = A[i]
                Segment_length = (i - preL)*2
                segment.append((preL,preL+Segment_length,L))
                preL = L
    
    for i in segment:
        (start_segment,end_segment,pitch) = i
        #result.extend([0]*(end_segment - start_segment))
        #print(result)
        X[start_segment:end_segment]
        length_segment = len(end_segment - start_segment)
        quater = length_segment / 4
        
    return (result)
                

def speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]

def stretch(sound_array, f, window_size, h):
    """ Stretches the sound by a factor `f` """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( len(sound_array) /f + window_size)

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):

        # two potentially overlapping subarrays
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # add to result
        i2 = int(i/f)
        result[i2 : i2 + window_size] += (hanning_window*a2_rephased).real

    result = ((2**(16-2)) * result/result.max()) # normalize (16bit)

    return result.astype('int16')
    
def pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)


'''



#result for problem1A:
X=makeSignal([(783.991,1.0,0)],0.05)
pitchDetector(X)


#result for problem1B:
Spectrums = makeSpectrum("clarinet",freq=261.6)
X=makeSignal(Spectrums,0.05)
#displaySignal(X)
pitchDetector(X)

#result for problem1C:
X=readWaveFile("clarinet.wav")
displaySignal(X)
writeWaveFile("TianyangBefore.wav", X)
StretchedFile = pitchshift(X, 0.5, window_size=2**13, h=2**11)
displaySignal(StretchedFile)
writeWaveFile("TianyangAfter.wav", StretchedFile)


#result for problem1d:
X=readWaveFile("Trumpet_01.wav")
X = X[int(0.2*SR):int(0.2*SR+0.05*SR)]
pitchDetector(X)

#result for problem1E:
X=readWaveFile("Genesis01.wav")
X = X[int(12.1*SR):int(12.1*SR+0.05*SR)]
pitchDetector(X)

#result for problem2A:
X=makeSignal([(783.991,1.0,0)],0.05)
pitchDetector2(X)

#result for problem2B:
Spectrums = makeSpectrum("clarinet",freq=261.6)
X=makeSignal(Spectrums,0.05)
#displaySignal(X)
pitchDetector2(X)


#result for problem2C:
X=readWaveFile("SteelString.wav")
X = X[1*SR:int(1*SR+0.05*SR)]
pitchDetector2(X)

#result for problem2D:
X=readWaveFile("Trumpet_01.wav")
X = X[int(0.2*SR):int(0.2*SR+0.05*SR)]
pitchDetector2(X)

#result for problem2E:
X=readWaveFile("Genesis01.wav")
X = X[int(12.1*SR):int(12.1*SR+0.05*SR)]
pitchDetector2(X)
'''

#final project test out:
'''
X=readWaveFile("clarinet.wav")
displaySignal(X)
writeWaveFile("TianyangBefore.wav", X)
StretchedFile = pitchshift(X, -2)
displaySignal(StretchedFile)
writeWaveFile("TianyangAfter.wav", StretchedFile)  
'''



def parse_arguments():
    description = ('Use your computer keyboard as a "piano"')

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--wav', '-w',
        metavar='FILE',
        type=argparse.FileType('r'),
        default='SteelString.wav',
        help='WAV file (default: SteelString.wav)')
    parser.add_argument(
        '--keyboard', '-k',
        metavar='FILE',
        type=argparse.FileType('r'),
        default='typewriter2.kb',
        help='keyboard file (default: typewriter2.kb)')
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='verbose mode')

    return (parser.parse_args(), parser)

def main():
    # Parse command line arguments
    (args, parser) = parse_arguments()

    # Enable warnings from scipy if requested
    if not args.verbose:
        warnings.simplefilter('ignore')

    #print type(args.wav.name)
    #fps, sound = wavfile.read(args.wav.name)
    fps, sound = wavfile.read('SteelString.wav')

    tones = range(-25, 25)
    sys.stdout.write('Transponding sound file... ')
    sys.stdout.flush()
    transposed_sounds = [pitchshift(sound, n) for n in tones]  # Change the pitch of the sound with the (-25, 25) tones
    print('DONE')

    # So flexible ;)
    pygame.mixer.init(fps, -16, 1, 2048)
    # For the focus
    screen = pygame.display.set_mode((150, 150))

    keys = args.keyboard.read().split('\n')
    print(keys)
    sounds = map(pygame.sndarray.make_sound, transposed_sounds)
    key_sound = dict(zip(keys, sounds))

    is_playing = {k: False for k in keys}

    while True:
        event = pygame.event.wait()

        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
            key = pygame.key.name(event.key)

        if event.type == pygame.KEYDOWN:
            if (key in key_sound.keys()) and (not is_playing[key]):
                key_sound[key].play(fade_ms=50)
                is_playing[key] = True

            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                raise KeyboardInterrupt

        elif event.type == pygame.KEYUP and key in key_sound.keys():
            # Stops with 50ms fadeout
            key_sound[key].fadeout(50)
            is_playing[key] = False


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Goodbye')