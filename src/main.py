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
              markeredgecolor='k', markerfacecolor='g', markevery=5,label='nuly')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r', markevery=5,label='póly')

    # set the ticks
    r = 1.2; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    plt.gca().set_xlabel('Realná zložka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginárna zložka $\mathbb{I}\{$z$\}$')
    plt.gca().set_title(u"Butterworth Filter " + str(FilterFreq) + 'Hz')
    plt.legend(loc='upper right')


def Fcharasteticstics(argh, FilterNumber):
    _, ax = plt.subplots(1, 2, figsize=(12,4))

    ax[0].plot(w / 2 / np.pi * frameRate, np.abs(argh))
    ax[0].set_xlabel('Frekvencia [Hz]')
    ax[0].set_title('Modul frekvenčnej charakteristiky $|H(e^{j\omega})|$ Filter ' + str(FilterNumber))

    ax[1].plot(w / 2 / np.pi * frameRate, np.angle(argh))
    ax[1].set_xlabel('Frekvencia [Hz]')
    ax[1].set_title('Argument frekvenčnej charakteristiky $\mathrm{arg}\ H(e^{j\omega})$ Filter ' + str(FilterNumber))

    for ax1 in ax:
        ax1.grid(alpha=0.5, linestyle='--')

    plt.savefig('src/img/task9Filter' + str(FilterNumber) + '.pdf')
    plt.close()

def ImpulseResponse(b,a,FilterNumber):
    h = lfilter(b, a, imp)
    plt.figure(figsize=(7,5))
    plt.stem(np.arange(N_imp), h, basefmt=' ')
    plt.gca().set_xlabel('$n$')
    plt.gca().set_title('Impulzívna odozva $h[n]$ Filtru ' + str(FilterNumber))

    plt.grid(alpha=0.5, linestyle='--')
    plt.savefig('src/img/task7filter' + str(FilterNumber) + '.pdf')
    plt.close()


def SpectogramOfCosinus():
    
    audioData = wavfile.read('audio/4cos.wav')

    audioInArray = np.array(audioData[1], dtype=float)

    audioInArray -= np.mean(audioInArray)
    audioInArray /= np.abs(audioInArray).max()

    audioInArray = audioInArray [0:recordingTotalFrames]

    f, t, sgr = signal.spectrogram(audioInArray,frameRate,nperseg=1024, noverlap=512)
    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
    sgr_log = 10 * np.log10(sgr+1e-20) 
    plt.figure(figsize=(9,5))
    plt.pcolormesh(t,f,sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
    plt.savefig('src/img/task6CosSpectogram.pdf')
    plt.close()

def MyDFT(signalData):
    n = len(signalData)
    return [sum((signalData[k]*cmath.exp(-1j*2*cmath.pi*i*k/n) for k in range(n)))
            for i in range(n)]

# rozdelenie na ramce    
def GetFrames(signalData,frameLength):
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

########################################### MAIN ########################################### 


#####################################  1. Load Audio files #########################################
print("\nTask1\n**********\n")
print("AudioFile values:")

with contextlib.closing(wave.open(recording,'r')) as f:
    recordingTotalFrames = f.getnframes()
    frameRate = f.getframerate()
    print("Frames : " + str(recordingTotalFrames))

audioData = wavfile.read(recording)
audioInArray = np.array(audioData[1], dtype=float)
audioInArray = audioInArray [0:recordingTotalFrames]

duration = np.arange(audioInArray.size) / frameRate
print("Duration : " + (str(round(max(duration), 6)) + "s"))
plt.figure(figsize=(8,4))
plt.plot(duration, audioInArray)

plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukový signál')

plt.savefig('src/img/task1RawAudioFile.pdf')
plt.close()   

#################################  2. Preparation + Frames  #####################################

print("Maximal Value : " + str(max(audioInArray)))
print("Minimal Value : " + str(min(audioInArray)))

audioInArray -= np.mean(audioInArray)
audioInArray /= np.abs(audioInArray).max()
print("\nTask2\n**********\n")
print("Normalized Maximal Value : " + str(max(audioInArray)))
print("Normalized Maximal Value : " + str(min(audioInArray)))

audioInArray = audioInArray [0:recordingTotalFrames]
frames = GetFrames(audioInArray, 1024)

timeOfOneFrame = np.arange(frames[43].size) / frameRate


plt.figure(figsize=(10,5))
plt.plot(timeOfOneFrame,frames[42]) 
plt.gca().set_xlabel(u"$t[s]$")
plt.gca().set_title(u"ramec - recording")
plt.savefig('src/img/task2NormalizedFrame.pdf')
plt.close()



#####################################  3. DFT  #########################################

print("\nTask3\n**********\n")
MyDFTFrames= MyDFT(frames[43])
LibraryDFTFrames = np.fft.fft(frames, 1024)
dfts_equal = np.allclose(MyDFTFrames,LibraryDFTFrames[43])
print( "Are DFTs equal ? : {} " .format('Yes' if dfts_equal else 'No'))

plt.figure(figsize=(10,5))
plt.plot(np.real(MyDFTFrames[0:512]))
plt.gca().set_ylabel(u"$f[Hz]$") 
plt.gca().set_xlabel(u"$vzorky[n]$")
plt.gca().set_title(u"DFT - recording")
plt.savefig('src/img/task3MyDFT.pdf')
plt.close()

fDFT = np.arange(LibraryDFTFrames[43][0:512].size) / 1024 * frameRate

plt.figure(figsize=(10,5))
plt.plot(fDFT,np.real(LibraryDFTFrames[43][0:512]))
plt.gca().set_xlabel(u"$f[Hz]$")
plt.gca().set_title(u"DFT - recording")
plt.savefig('src/img/task3LibDFT.pdf')
plt.close()
shape = len(frames)


#####################################  4. Spectogram   ######################################

f, t, sgr = signal.spectrogram(audioInArray,frameRate,nperseg=1024, noverlap=512)
# prevod na PSD
# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
sgr_log = 10 * np.log10(sgr+1e-20) 
plt.figure(figsize=(9,5))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.savefig('src/img/task4Spectrum.pdf')
plt.close()

#####################################  5. Generating Signals  ######################################

f1 = 965
f2 = 2*f1
f3 = 3*f1
f4 = 4*f1

#####################################  6. Cosinuses Spectogram ######################################

fourCos = []
for i in range(recordingTotalFrames):
    fourCos.append(i/frameRate)

outputCos1 = np.cos(2 * np.pi * f1 * np.array(fourCos))
outputCos2 = np.cos(2 * np.pi * f2 * np.array(fourCos))
outputCos3 = np.cos(2 * np.pi * f3 * np.array(fourCos))
outputCos4 = np.cos(2 * np.pi * f4 * np.array(fourCos))

outputCos = outputCos1 + outputCos2 + outputCos3 + outputCos4
wavfile.write("audio/4cos.wav",frameRate,outputCos.astype(np.float32))


SpectogramOfCosinus();

#####################################  7. Generate Filters  ######################################

print("\nTask7\n**********\n")
nyq = 0.5 * frameRate
lowStop = (f1-30) / nyq
highStop = (f1+30) / nyq
lowPass = (f1-80) / nyq
highPass = (f1+80) / nyq
N,Wn = signal.buttord([lowPass, highPass], [lowStop, highStop], 3, 40)
b1, a1 = signal.butter(N, Wn, btype='bandstop')
w, h1 = signal.freqz(b1, a1)
plt.plot(w/np.pi*frameRate/2, 20 * np.log10(abs(h1)))
print("Filter 1 coefficients:\n\nA\n")
print(a1)
print("\nB\n")
print(b1)


lowStop = (f2-30) / nyq
highStop = (f2+30) / nyq
lowPass = (f2-80) / nyq
highPass = (f2+80) / nyq
N,Wn = signal.buttord([lowPass, highPass], [lowStop, highStop], 3, 40)
b2, a2 = signal.butter(N, Wn, btype='bandstop')
w, h2 = signal.freqz(b2, a2)
plt.plot(w/np.pi*frameRate/2, 20 * np.log10(abs(h2)))
print("\nFilter 2 coefficients:\n\nA\n")
print(a2)
print("\nB\n")
print(b2)


lowStop =  (f3-30) / nyq
highStop = (f3+30) / nyq
lowPass = (f3-80) / nyq
highPass = (f3+80) / nyq
N,Wn = signal.buttord([lowPass, highPass], [lowStop, highStop], 3, 40)
b3, a3 = signal.butter(N, Wn, btype='bandstop')
w, h3 = signal.freqz(b3, a3)
plt.plot(w/np.pi*frameRate/2, 20 * np.log10(abs(h3)))
print("\nFilter 3 coefficients:\n\nA\n")
print(a3)
print("\nB\n")
print(b3)

lowStop = (f4-30) / nyq
highStop = (f4+30) / nyq
lowPass = (f4-80) / nyq
highPass = (f4+80) / nyq
N,Wn = signal.buttord([lowPass, highPass], [lowStop, highStop], 3, 40)
b4, a4 = signal.butter(N, Wn, btype='bandstop')
w, h4 = signal.freqz(b4, a4)
plt.plot(w/np.pi*frameRate/2, 20 * np.log10(abs(h4)))
plt.title('Butterworth filter frequency response')
plt.gca().set_xlabel('f [hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.savefig('src/img/task9ModulsOfFilters.pdf')
plt.close()
print("\nFilter 4 coefficients:\n\nA\n")
print(a4)
print("\nB\n")
print(b4)

# impulsni odezva
N_imp = 32
imp = [1, *np.zeros(N_imp-1)]

ImpulseResponse(b1,a1,1)
ImpulseResponse(b2,a2,2)
ImpulseResponse(b3,a3,3)
ImpulseResponse(b4,a4,4)


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

#####################################  9. Response characteristic  ######################################

Fcharasteticstics(h1,1)
Fcharasteticstics(h2,2)
Fcharasteticstics(h3,3)
Fcharasteticstics(h4,4)

#####################################  10. Filtration  ######################################

sf = lfilter(b1, a1, audioInArray)
sf = lfilter(b2, a2, sf)
sf = lfilter(b3, a3, sf)
sf = lfilter(b4, a4, sf)
f, t, sfgr = signal.spectrogram(sf, frameRate)
sfgr_log = 10 * np.log10(sfgr+1e-20)

plt.figure(figsize=(10,4))
plt.pcolormesh(t,f,sfgr_log)
plt.gca().set_title('Spektrogram vyfiltrovaného signálu')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.savefig('src/img/task10FinalSignal.pdf')

 
sf -= np.mean(sf)
sf /= np.abs(sf).max()

print("\nTask10\n**********\n")
polite_state = (  max(sf) <= 1  and  min(sf) >= -1 )
print('Signal {} in polite state.\n'.format('is' if polite_state else 'is not'))


wavfile.write("audio/bandstop.wav",frameRate,sf.astype(np.float32))
