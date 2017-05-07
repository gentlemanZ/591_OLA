import pyaudio
import sys, time
import numpy as np
import wave

n = 3 # this is how the pitch should change, positive integers increase the frequency, negative integers decrease it.
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 41000
RECORD_SECONDS = 5
swidth = 2

p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                output = True,
                frames_per_buffer = chunk)


print ("* recording")

start = time.time()
while(time.time()-start < RECORD_SECONDS):

    data = stream.read(chunk)
    data = np.array(wave.struct.unpack("%dh"%(len(data)/swidth), data))

    # do real fast Fourier transform 
    data = np.fft.rfft(data)
    
    # This does the shifting
    data2 = [0]*len(data)
    if n >= 0:
        data2[n:len(data)] = data[0:(len(data)-n)]
        data2[0:n] = data[(len(data)-n):len(data)]
    else:
        data2[0:(len(data)+n)] = data[-n:len(data)]
        data2[(len(data)+n):len(data)] = data[0:-n]
    
    data = np.array(data2)
    # Done shifting
    
    # inverse transform to get back to temporal data
    data = np.fft.irfft(data)
    
    dataout = np.array(data, dtype='int16') 
    chunkout = wave.struct.pack("%dh"%(len(dataout)), *list(dataout)) #convert back to 16-bit data
    stream.write(chunkout)

    


print ("* done")

stream.stop_stream()
stream.close()
p.terminate()