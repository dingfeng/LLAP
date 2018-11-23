# -*- coding: UTF-8 -*-
# filename: pcm2wav date: 2018/11/22 12:54  
# author: FD 
import os
import queue
import wave

if __name__ == '__main__':
    queue = queue.Queue()
    rootDir = '.'
    queue.put(rootDir)
    while not queue.empty():
        node = queue.get()
        for filename in os.listdir(node):
            nextpath = os.path.join(node, filename)
            if os.path.isdir(nextpath):
                queue.put(nextpath)
            elif nextpath.endswith('.pcm'):
                filenamenosuffix = nextpath[0: nextpath.index('.pcm')]
                wavfilepath = filenamenosuffix + ".wav"
                with open(nextpath, 'rb') as pcmfile:
                    pcmdata = pcmfile.read()
                with wave.open(wavfilepath, 'wb') as wavfile:
                    wavfile.setparams((1, 2, 48000, 0, 'NONE', 'NONE'))
                    wavfile.writeframes(pcmdata)