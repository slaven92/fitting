import numpy as np
from tkinter.filedialog import askopenfilename
import tkinter
import os

import json


class Convert():
    def __init__(self, file_location = None, threshold=70):
        self.results = [] # array of fitted data. Name is contained in outParams
        self.file_location = file_location
        self.para_dic = [] # loaded name of vectors
        self.dataFull = [] # all data loaded in numpy array 
        self.outParams = [] # name of fitted parameters
        self.initParam = [] # parameters used to initilize the fitting
        self.results_averaged = [] # results from fitting of averaged data
        self.threshold = threshold

        if self.file_location == None:
            tkinter.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
            self.file_location = askopenfilename() # show an "Open" dialog box and return the path to the selected file
        
        self.loadFullData()
        self.dump()

    def loadFullData(self):
        oneMeasurement = []
        is_scaning_Comment = True
        with open(self.file_location,'r') as cmt_file:    # open file
            for line in cmt_file:    # read each line

                if is_scaning_Comment and len(line) == 1:
                    continue

                if is_scaning_Comment and '\tname: ' in line:
                    self.para_dic.append(line[8:-1])
                    continue

                if is_scaning_Comment and line[0] != '#':
                    is_scaning_Comment = False
                    oneMeasurement = [ [] for _ in self.para_dic]

                if not is_scaning_Comment:
                    if len(line) == 1:
                        self.dataFull.append(oneMeasurement)
                        oneMeasurement = [ [] for _ in self.para_dic]
                    else:
                        splited_line = line.split('\t')
                        for i, number in enumerate(splited_line):
                            oneMeasurement[i].append(float(number))
        
        self.dataFull = np.array(self.dataFull)

    def dump(self):
        self.data = []
        self.averagedData = []
        inner = []

        for i, mesur in enumerate(self.dataFull):
            #look at the temperatures
            curr_temp = mesur[1,0]
            last_temp = self.dataFull[i-1,1,0] if i!=0 else -1
            next_temp = self.dataFull[i+1,1,0] if i!=len(self.dataFull)-1 else -1


            # check if on first element with same temp. if yes, do interplation and continue
            if curr_temp!=last_temp:
                curr_averaged = np.interp(self.dataFull[i+1,0,:], mesur[0,:], mesur[2,:])
                curr_num_of_traces = 1
                inner = [list(curr_averaged)]
                continue
            

            #check if last element with same temp. if yes, save averaged and continue
            if next_temp!=curr_temp:
                curr_num_of_traces += 1
                curr_averaged = [old + (new - old)/curr_num_of_traces for old,new in zip(curr_averaged, mesur[2])]
                inner.append(list(mesur[2,:]))
                self.averagedData.append({'x':list(mesur[0,:]), 't':mesur[1,0], 'y':curr_averaged})
                # self.averagedData.append([mesur[0,:], mesur[1,:], curr_averaged])
                self.data.append({'x':list(mesur[0,:]),'t':mesur[1,0],'y':inner})
                continue
            
            #normal case do averaging
            curr_num_of_traces += 1
            curr_averaged = [old + (new - old)/curr_num_of_traces for old,new in zip(curr_averaged, mesur[2])]
            inner.append(list(mesur[2,:]))
        
        self.saveJson()
    
    def saveJson(self):
        dirname = os.path.dirname(self.file_location)
        filename = os.path.splitext(os.path.basename(self.file_location))[0]
        json_file_location = filename + '.json'
        json_file_location = os.path.join(dirname, json_file_location)
        json_avr_file_location = filename + '_avr.json'
        json_avr_file_location = os.path.join(dirname, json_avr_file_location)

        with open(json_file_location, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        with open(json_avr_file_location, 'w', encoding='utf-8') as f:
            json.dump(self.averagedData, f, ensure_ascii=False, indent=4)


# f4 = 'V:/20200403/114454_spectrum_vs_freq_and_temperature/114454_spectrum_vs_freq_and_temperature.dat'
f4 = Convert()
f4.dump()