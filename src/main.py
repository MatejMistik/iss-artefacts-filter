# coding=utf-8

# import libraries

import wave
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from scipy.io import wavfile
# Global variables



f16 = 16000

# functions 



########################################### MAIN ########################################### 




# Uloha 3.
# recording normalization

# get file duration
recording = 'audio/xmisti00.wav'
with contextlib.closing(wave.open(recording,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    print("duration " + str(duration))
    print("frames " + str(frames))
    print("framerate " + str(rate))



data_off = wavfile.read(recording)

array_off = np.array(data_off[1], dtype=float)
print("MAX bez normalizace " + str(max(array_off)))
print("MIN bez normalizace " + str(min(array_off)))

array_off -= np.mean(array_off)
array_off /= np.abs(array_off).max()
print(max(array_off))
print(min(array_off))
""""
array_off = array_off [5000:21000]
framer = np.zeros(shape=(99, 320))
for i in range(99):
    frame = array_off[(i*160):320+(i*160)]
    framer[i] = frame

task3_off = framer
Clip_off = framer
## print  #  
### mask off ###

plt.figure(figsize=(10,5))
plt.plot(task3_off[0]) 
plt.gca().set_xlabel(u"$t[s]$")
plt.gca().set_title(u"ramec - recording")


data_on = wavfile.read(recording)
array_on = np.array(data_on[1], dtype=float)
array_on -= np.mean(array_on)
array_on /= np.abs(array_on).max()
array_on = array_on [5000:21000]
framer = np.zeros(shape=(99, 320))
for i in range(99):
    frame = array_on[(i*160):320+(i*160)]
    framer[i] = frame



task3_on = framer
Clip_on = framer

## print ##
### mask on ###
plt.figure(figsize=(10,5))
plt.plot(task3_on[42]) 
plt.gca().set_xlabel(u"$t[s]$")
plt.gca().set_title(u"ramec - recording_on")



################################################ task 4 ########################################

# clip_off = framer [0:99][0:320]

### maskoff 4 ###

up_max = Clip_off.max()*float(0.7)
up_min = Clip_off.min()*float(0.7)

clipped_maskoff = []
for g in range (99):
    clip = []
    for i in range (320):
        if Clip_off[g][i] > up_max :
          clip.append(1)
        elif Clip_off[g][i] < up_min :
            clip.append(-1)
        else:
            clip.append(0)
    clipped_maskoff.append(clip)

plt.figure(figsize=(10,5))
plt.plot(clipped_maskoff[4]) 
plt.gca().set_xlabel(u"$t[s]$")
plt.gca().set_title(u"Center Clipping - recording")

#### autokolerace ###
   ## clipped_maskoff [99][320] clipped frames
koler = []
for g in range (99):
    koler_clips = []
    for k in range (319): ## k
        suma = 0
        for n in range (319-1-k):  ## n  
            suma +=(clipped_maskoff[g][n]*(clipped_maskoff[g][n+k]))
        koler_clips.append(suma)
    koler.append(koler_clips)

plt.figure(figsize=(10,5))
plt.axvline(x=5)
plt.plot(koler[0])  
plt.gca().set_xlabel(u"$t[s]$")
plt.gca().set_title(u"autokolerace - recording")


### lag ###

lag_index = []
for i in range (99):
    lag = np.argmax(koler[i][30:])+30 # lag
    f0_off = f16/lag
    lag_index.append(f0_off)

 ### maskon 4 ###
up_max = Clip_on.max()*float(0.7)
up_min = Clip_on.min()*float(0.7)

clipped_maskon = [] ## will include clips of [99][320]
for g in range (99):
    clip = []
    for i in range (319):
        if Clip_on[g][i] > up_max :
          clip.append(1)
        elif Clip_on[g][i] < up_min :
            clip.append(-1)
        else:
            clip.append(0)
    clipped_maskon.append(clip)    

plt.figure(figsize=(10,5))
plt.plot(clipped_maskon[4]) 
plt.gca().set_xlabel(u"$t[s]$")
plt.gca().set_title(u"Center Clipping - recording_on")

koler = []
for g in range (99):
    koler_clips = []
    for k in range (319): ## k
        suma = 0
        for n in range (319-1-k):  ## n  
            suma +=(clipped_maskon[g][n]*(clipped_maskon[g][n+k]))
        koler_clips.append(suma)
    koler.append(koler_clips)

plt.figure(figsize=(10,5))
plt.axvline(x=5)
plt.plot(koler[0])  
plt.gca().set_xlabel(u"$t[s]$")
plt.gca().set_title(u"autokolerace - recording_on")
plt.show()

"""





