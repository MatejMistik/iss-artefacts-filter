# coding=utf-8

# Matej Mistik

# import libraries
import soundfile as sf
from scipy import signal
from scipy.signal import tf2zpk
import wave
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
from scipy.signal import lfilter
from scipy.io import wavfile
import cmath


# Global variables
recording = 'audio/xmisti00.wav'

# functions
"""
***************************************************************************************/
*    Title: PYTHON ZPLANE FUNCTION
*    Author: Christopher Felton
*    Date: 17.12., 2011
*    Code version: 1.0
*    Availability: https://www.dsprelated.com/showcode/244.php
*
***************************************************************************************/
"""
def zplane(b,a,placeNumber,FilterFreq):
    """Plot the complex z-plane given a transfer function.
    """
    
    # get a figure/plot
    ax = plt.subplot(1,2,placeNumber)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='deepskyblue')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g',label='nuly')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r',label='póly')

    # set the ticks
    r = 1.2; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    plt.gca().set_xlabel('Realná zložka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginárna zložka $\mathbb{I}\{$z$\}$')
    plt.gca().set_title(u"Butterworth Filter " + str(FilterFreq) + 'Hz')
    plt.legend(loc='upper right')

    return z, p, k

def Task6():
    
    print("****Generating task 6 ****")

    audioData = wavfile.read('audio/4cos.wav')

    audioInArray = np.array(audioData[1], dtype=float)
    print("MAX bez normalizace " + str(max(audioInArray)))
    print("MIN bez normalizace " + str(min(audioInArray)))

    audioInArray -= np.mean(audioInArray)
    audioInArray /= np.abs(audioInArray).max()
    print(max(audioInArray))
    print(min(audioInArray))

    audioInArray = audioInArray [0:recordingTotalFrames]

    f, t, sgr = signal.spectrogram(audioInArray,frameRate,nperseg=1024, noverlap=512)
    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
    sgr_log = 10 * np.log10(sgr+1e-20) 
    plt.figure(figsize=(9,3))
    plt.pcolormesh(t,f,sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig('src/img/task6.pdf')
    plt.close()

def myDFT(signalData):
    n = len(signalData)
    return [sum((signalData[k]*cmath.exp(-1j*2*cmath.pi*i*k/n) for k in range(n)))
            for i in range(n)] 

########################################### MAIN ########################################### 


#####################################  1. Load Audio files #########################################

with contextlib.closing(wave.open(recording,'r')) as f:
    recordingTotalFrames = f.getnframes()
    frameRate = f.getframerate()
    duration = recordingTotalFrames / float(frameRate)
    print("duration " + str(duration))
    print("recordingTotalFrames " + str(recordingTotalFrames))
    print("framerate " + str(frameRate))

# rozdelenie na ramce    
def getFrames(signalData,frameLength):
    length=len(signalData) 
    overlap= 512
    framesize=int(frameLength)
    countFrames= int(length/overlap);
    frames=np.ndarray((countFrames,framesize)) 
    for k in range(0,countFrames):
        for i in range(0,framesize):
            if((k*overlap+i)<length):
                frames[k][i]=signalData[k*overlap+i]
            else:
                frames[k][i]=0
    return frames

audioData = wavfile.read(recording)
#################################  2. Preparation + Frames  #####################################

audioInArray = np.array(audioData[1], dtype=float)
print("MAX bez normalizace " + str(max(audioInArray)))
print("MIN bez normalizace " + str(min(audioInArray)))

audioInArray -= np.mean(audioInArray)
audioInArray /= np.abs(audioInArray).max()
print(max(audioInArray))
print(min(audioInArray))

audioInArray = audioInArray [0:recordingTotalFrames]
frames = getFrames(audioInArray, 1024)

timeOfOneFrame = np.arange(frames[43].size) / frameRate

"""
plt.figure(figsize=(10,5))
plt.plot(timeOfOneFrame,frames[42]) 
plt.gca().set_xlabel(u"$t[s]$")
plt.gca().set_title(u"ramec - recording")
plt.savefig('src/img/task2.pdf')
plt.close()
"""


#####################################  3. DFT  #########################################


#MyDFTFrames= myDFT(frames[42])
LibraryDFTFrames = np.fft.fft(frames, 1024)
#print( "Are DFTs equalt? : "+ str(np.allclose(MyDFTFrames,LibraryDFTFrames)))

"""
plt.figure(figsize=(10,5))
plt.plot(np.real(MyDFTFrames[0:512]))
plt.gca().set_ylabel(u"$f[Hz]$") 
plt.gca().set_xlabel(u"$vzorky[n]$")
plt.gca().set_title(u"DFT - recording")
plt.savefig('src/img/task3MyDFT.pdf')
plt.close()
"""
fDFT = np.arange(LibraryDFTFrames[43][0:512].size) / 1024 * frameRate

plt.figure(figsize=(10,5))
plt.plot(fDFT,np.real(LibraryDFTFrames[43][0:512]))
plt.gca().set_xlabel(u"$f[Hz]$")
plt.gca().set_title(u"DFT - recording")
plt.savefig('src/img/task3LibDFT.pdf')
plt.close()
shape = len(frames)
print(frames.size)
print(shape)
#####################################  4. Spectogram   ######################################

spectrum = np.ndarray((shape,512), dtype=np.complex128)

for k in range(0,shape):
    spectrum[k] = LibraryDFTFrames[k][0:512]

N = frames.size
# vykreslenie spektogramu
f, ax = plt.subplots(figsize=(10, 3))
S = np.abs(spectrum)
S = 10 * np.log10(1/N * S**2)
ax.set_title('Spektogram bez rúška')
cax = ax.imshow(np.rot90(S), cmap='viridis', aspect='auto', extent=[0, duration, 0, 8000])
ax.axis('auto')
ax.set_ylabel('Frekvencia [Hz]')
ax.set_xlabel('Čas [s]');
cbar   = f.colorbar(cax,aspect=10)
cbar.set_label('Spektrálna hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig('src/img/task4Spectrum.pdf')
plt.close()

"""
f, t, sgr = signal.spectrogram(audioInArray,frameRate,nperseg=1024, noverlap=512)
# prevod na PSD
# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
sgr_log = 10 * np.log10(sgr+1e-20) 
plt.figure(figsize=(9,3))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()
"""

#####################################  5+6. Generating Signals  ######################################

## in hz need to convert for 16000hz framerate
f1 = 976
f2 = 2*f1
f3 = 3*f1
f4 = 4*f1

fourCos = []
for i in range(recordingTotalFrames):
    fourCos.append(i/frameRate)

outputCos1 = np.cos(2 * np.pi * f1 * np.array(fourCos))
outputCos2 = np.cos(2 * np.pi * f2 * np.array(fourCos))
outputCos3 = np.cos(2 * np.pi * f3 * np.array(fourCos))
outputCos4 = np.cos(2 * np.pi * f4 * np.array(fourCos))

outputCos = outputCos1 + outputCos2 + outputCos3 + outputCos4
wavfile.write("audio/4cos.wav",frameRate,outputCos.astype(np.float64))

Task6();

#####################################  7. Generate Filters  ######################################


nyq = 0.5 * frameRate
low = (f1-15) / nyq
high = (f1+15) / nyq
b1, a1 = signal.butter(4, [low,high], btype='bandstop')
w, h = signal.freqz(b1, a1)
plt.plot(w/np.pi*frameRate/2, 20 * np.log10(abs(h)))



low = (f2-15) / nyq
high = (f2+15) / nyq
b2, a2 = signal.butter(4, [low,high], btype='bandstop')
w, h = signal.freqz(b2, a2)
plt.plot(w/np.pi*frameRate/2, 20 * np.log10(abs(h)))



low =  (f3-15) / nyq
high = (f3+15) / nyq
b3, a3 = signal.butter(4, [low,high], btype='bandstop')
w, h = signal.freqz(b3, a3)
plt.plot(w/np.pi*frameRate/2, 20 * np.log10(abs(h)))


low = (f4-15) / nyq
high = (f4+15) / nyq
b4, a4 = signal.butter(4, [low,high], btype='bandstop')
w, h = signal.freqz(b4, a4)

plt.plot(w/np.pi*frameRate/2, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.gca().set_xlabel('f [hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.savefig('src/img/task7filters.pdf')
plt.close()


#####################################  8. Poles and Zeros  ######################################
plt.figure(figsize=(10,5))
zplane(b1,a1,1,f1)
zplane(b2,a2,2,f2)
plt.savefig('src/img/task8polesA.pdf')
plt.close()
plt.figure(figsize=(10,5))
zplane(b3,a3,1,f3)
zplane(b4,a4,2,f4)
plt.savefig('src/img/task8polesB.pdf')
plt.close()

#####################################  9. Poles and Zeros  ######################################
