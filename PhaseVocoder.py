

# phase vocoder example
# (c) V Lazzarini, 2010
# GNU Public License
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



from scipy import *
from pylab import *
from scipy.io import wavfile

#import cs591Utilities as au
def PhaseVocoderFunc(X,P):
    N = 4096      # window size for FFT
    H = N//4       # skip size -- how far the window advances at each step
    
    
    tscale = 1.0/P
    
    # signal blocks for processing and output
    phi  = zeros(N)
    out = zeros(N, dtype=complex)
    Y = zeros(int(len(X)/tscale+N))
    
    # max input amp, window
    amp = max(X)
    win = hanning(N)
    
    # p is beginning of window
    p = 0
    pp = 0
    
    
    while(p < len(X)-(N+H)):
    #    print(str(p) + "\t" + str(pp))
    
        # take the spectra of two consecutive windows
        spec1 =  fft(win*X[p:p+N])
        spec2 =  fft(win*X[p+H:p+N+H])
        
        # take their phase difference and integrate
        phi += (angle(spec2) - angle(spec1))
        
        # bring the phase back to between pi and -pi
        for i in range(len(phi)):
            while(phi[i] < -pi): 
                phi[i] += 2*pi
            while(phi[i] >= pi): 
                phi[i] -= 2*pi
        out.real, out.imag = cos(phi), sin(phi)
        
        # inverse FFT and overlap-add
        Y[pp:pp+N] += (win*ifft(spec2*out)).real
    
        
        pp += H
        p += int(H*tscale)
    
    
    #  write file to output, scaling it to original amp
    return array(amp*Y/max(Y))

