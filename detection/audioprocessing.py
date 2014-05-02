import os
import requests
import json
import sys
import shutil
import numpy as np
import scipy
from scipy.io.wavfile import read, write
from scipy.signal import butter, lfilter


class AudioAPI(object):
    def __init__(self):
        self.events = list()
        self.tags = ['jagged', 'tagged', 'lagged', 'f*****', 'tag it', 
                    'tag', 'target', 'had it', 'bag it','Tegan', 'Sagat'
                    'Saget', 'Tagetes', 'agate', 'Faget', 'lag it', 'bag', 
                    'lag', 'sag', 'sag it', 'had', 'Daggett', 'Zagat', 'dagger']
        self.watches = ["watch this", "wash this", "what's this", "what's",
                        "watch", "wash"]
        self.eggf = ['f***']
        self.eggs =  ['s***']

    def create_event(self, **kwargs):
        self.events.append(kwargs)

    def cutmeopen_mom(self,sig,fs,winsize):
        trshu = 0.1
        trshs = 0.015
        winstep = int(winsize/3)
        numwin = int((len(sig) - winsize)/winstep)
        tea = self.get_en(sig,fs,winsize)
        normtea = self.normalize(tea)
        utt, butt = self.first_chop(normtea,trshu)
        sutt, sbutt = self.other_chop(normtea,trshs)
        return sutt, sbutt, utt, butt

    def process(self, sig,fs):
        words = list()
        responses = list()
        emph = self.preemphasis(sig)
        filtered = self.butter_bandpass_filter(emph,250,3500,fs,6)
        winsize = int(0.03*fs)
        winstep = int(winsize/3)
        sbeg, send, beg, end = self.cutmeopen_mom(sig,fs,winsize)
        for i in range(0,len(beg)):
            if((end[i]-beg[i])>110):
                time = self.gettime(beg[i],winstep,fs)
                timestamp =  (np.floor(time*10))/10
                self.create_event(event_type="Tag",
                            event_subtype="Loud",
                            timestamp=timestamp)
        buff_beg, buff_end = self.get_buffers(sbeg,send)
        self.clean_temp()
        for i in range(0,len(buff_beg)):
            st = self.getsamp(buff_beg[i],winsize)
            fin = self.getsamp(buff_end[i],winsize) + winsize
            if len(sig[st:fin] > 44):
                self.write_tempwav('tempo_'+str(i)+'.wav',fs,sig[st:fin])
                self.convert2flac(('tempo_'+str(i)+'.wav'))
        
        for i in range(0,len(buff_beg)):
            if (buff_end[i]-buff_beg[i]):
                responses.append(self.detect('temp/tempo_'+str(i)+'.flac'))

        tags, watches, fck, sht = self.check_responses(responses,buff_beg)
        tags = self.debounce(tags)
        watches = self.debounce(watches)
        fck = self.debounce(fck)
        sht = self.debounce(sht)
        for tag in tags:
            time = self.gettime(tag,winstep,fs)
            timestamp =  (np.floor(time*10))/10
            self.create_event(event_type="Tag",
                        event_subtype="Tag",
                        timestamp=timestamp)
        for wat in watches:
            time = self.gettime(wat,winstep,fs)
            timestamp =  (np.floor(time*10))/10
            self.create_event(event_type="Tag",
                        event_subtype="Watch",
                        timestamp=timestamp)
        for f in fck:
            time = self.gettime(f,winstep,fs)
            timestamp =  (np.floor(time*10))/10
            self.create_event(event_type="Tag",
                        event_subtype="Egg",
                        timestamp=timestamp)
        for sh in sht:
            time = self.gettime(sh,winstep,fs)
            timestamp =  (np.floor(time*10))/10
            self.create_event(event_type="Tag",
                        event_subtype="Egg",
                        timestamp=timestamp)
        return self.events

    def write_tempwav(self, filename, sampling_freq, data):
        # Imported from scipy.io.wavfile
        final_path = "temp/%s" % (filename)
        write(final_path, sampling_freq, data)

    def convert2flac(self,filename):
        FLAC_CONV = 'flac -f -s --delete-input-file --sample-rate=16000'
        os.system(FLAC_CONV+' '+'temp/'+filename)

    def clean_temp(self):
        path = 'temp'
        if os.path.exists(path) :
            shutil.rmtree(path)
        os.mkdir(path)

    def check_responses(self,responses, buff_beg):
        tag = list()
        watch = list()
        fs = list()
        sh = list()
        for i in xrange(len(responses)):
            if responses[i] and responses[i].text:
                info = eval(responses[i].text)
                for watches in self.watches:
                    if watches in responses[i].text:
                        watch.append(buff_beg[i])
                for tagit in self.tags:
                    if tagit in responses[i].text:
                        tag.append(buff_beg[i])
                for egf in self.eggf:
                    if egf in responses[i].text:
                        fs.append(buff_beg[i])
                for egs in self.eggs:
                    if egs in responses[i].text:
                        sh.append(buff_beg[i])
        return tag, watch, fs, sh

    def debounce(self,keyword):
        trash = []
        for i in range(0,len(keyword)-1):
            if (keyword[i+1]-keyword[i])<=150:
                trash.append(keyword[i])
        for i in trash:
            keyword.remove(i)
        return keyword

    def detect(self,input):
        try:
            current_path = os.getcwd()
            definite_path = os.path.join(current_path,input)
            flac = {'file' : open(definite_path)}
            headers = {'Content-type':'audio/x-flac;rate=16000'}
            language = 'en-US'
            parameters = {'xjerr':'1','client':'chromium','lang':language}
            url = 'https://www.google.com/speech-api/v1/recognize'
            req = requests.post(url,
                                params=parameters,
                                headers=headers,
                                files=flac)
            print req.text
            return req
        except:
            return None

    def get_buffers(self,begs,ends):
        buff_beg = []
        buff_end = []
        for j in range(0,len(begs)):
            buff_beg.append(begs[j])
            for i in range(j,len(ends)):
                if((ends[i]-buff_beg[j])>150):
                    buff_end.append(ends[i-1])
                    break
                elif (ends[i] == ends[len(ends)-1]):
                    buff_end.append(ends[i])
                    break
        return buff_beg, buff_end

    def getwavdata(self, file):
        rate, data = scipy.io.wavfile.read(file)
        return rate, data

    def whichone(self, sig):
        r = sig[:,0]
        l = sig[:,1]
        enr = np.float64(0)
        enl = np.float64(0)
        for i in range(0,int(len(r)/8)):
            enr += r[i]**2
            enl += l[i]**2
        if (enr>enl):
            return 1
        elif (enr<=enl):
            return 0

    def butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order)
        y = scipy.signal.lfilter(b, a, data)
        return y

    def gettime(self, win,winstep,fs):
        time = np.float64(win*winstep)
        return (time/fs)

    def getsamp(self, win,winsize):
        samp = win*(int(winsize/3))
        return samp

    def tossme(self, begs,ends):
        utts = len(begs)
        trashb = []
        trashe = []
        for i in range(0,utts):
            if((ends[i]-begs[i])<3):
                trashb.append(begs[i])
                trashe.append(ends[i])
        for i in range(0,len(trashe)):
            begs.remove(trashb[i])
            ends.remove(trashe[i])
        return begs, ends

    def preemphasis(self, signal,coeff=0.95):
        return np.append(signal[0],signal[1:]-coeff*signal[:-1])

    def normalize(self, energy):
        maxi = np.max(energy)
        mini = np.min(energy)
        norm = []
        for i in range(0,len(energy)):
            norm.append((energy[i]-mini)/(maxi-mini))
        return norm

    def get_ens(self, sig,fs,winsize):
        winstep = int(winsize/3)
        numwin = int((len(sig) - winsize)/winstep)
        ham = scipy.signal.hamming(winsize)
        tea = []
        ent = []
        en = []
        eef = []
        for i in range(0,numwin):
            winst = (i*winstep)
            winen = (i*winstep) + winsize
            winps = np.abs(np.fft.fft((sig[winst:winen])*ham))**2
            f = []
            for j in range(0,winsize):
                f.append((j**2)*winps[j])
            tea.append(0)
            for k in range(0,winsize):
                tea[i] += f[k]
            tea[i] = np.sqrt(tea[i])
            en.append(np.float64(0))
            pdf = []
            sumps = 0
            ent.append(0)
            for j in range(0,winsize):
                en[i] += (sig[winst+j])**2
                sumps += winps[j]
            for k in range(0,winsize):
                pdf.append((winps[k]/sumps))
                ent[i] += pdf[k]*np.log(pdf[k])
            eef.append(np.sqrt(1+ np.abs(ent[i]*en[i])))
        return (tea, eef)

    def get_en(self,sig,fs,winsize):
        winstep = int(winsize/3)
        numwin = int((len(sig) - winsize)/winstep)
        ham = scipy.signal.hamming(winsize)
        tea = []
        for i in range(0,numwin):
            winst = (i*winstep)
            winen = (i*winstep) + winsize
            winps = np.abs(np.fft.fft((sig[winst:winen])*ham))**2
            f = []
            for j in range(0,winsize):
                f.append((j**2)*winps[j])
            tea.append(0)
            for k in range(0,winsize):
                tea[i] += f[k]
            tea[i] = np.sqrt(tea[i])
        return tea   


    def first_chop(self, normtea,trshu):
        check = True
        numwin = len(normtea)
        utt = []
        butt = []
        for i in range(0,numwin):
            if(check):
                if(normtea[i]>=trshu):
                    utt.append(i)
                    check = False
            else:
                if(normtea[i]<trshu):
                    butt.append(i)
                    check = True
        if(len(utt)!=len(butt)):
            utt.remove(utt[len(utt)-1])
        nutt ,nbutt = self.merge(utt,butt)
        finutt, finbutt = self.tossme(nutt,nbutt)
        return (finutt, finbutt)

    def merge(self,begs1,ends1):
        garb1 = []
        gare1 = []
       
        # MERGE UTTERANCES THAT ARE TOO CLOSE
        for i in range(0,len(begs1)-1):
           
            if((begs1[i+1]-ends1[i])<12):
     
                garb1.append(begs1[i+1])
                gare1.append(ends1[i])
               
           
        for i in range(0,len(garb1)):
            begs1.remove(garb1[i])
            ends1.remove(gare1[i])
     
        return begs1, ends1
           
    def tossme(self,begs1,ends1):
        trashb1 = []
        trashe1 = []
       
        # REMOVE UTTERANCES THAT ARE TOO SMALL
        for i in range(0,len(begs1)):
            if((ends1[i]-begs1[i])<12):
                trashb1.append(begs1[i])
                trashe1.append(ends1[i])
               
        for i in range(0,len(trashe1)):
            begs1.remove(trashb1[i])
            ends1.remove(trashe1[i])
           
       
        return begs1, ends1

    def other_chop(self,normtea,trshu):
        check = True
        numwin = len(normtea)
        utt = []
        butt = []
        for i in range(0,numwin):
            if(check):
                if(normtea[i]>=trshu):
                    utt.append(i)
                    check = False
            else:
                if(normtea[i]<trshu):
                    butt.append(i)
                    check = True
        if(len(utt)!=len(butt)):
            utt.remove(utt[len(utt)-1])
        return (utt, butt)
