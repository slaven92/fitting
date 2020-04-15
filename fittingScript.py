import numpy as np
from lmfit.models import ConstantModel, LorentzianModel
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tkinter
import os
import json
from scipy.interpolate import griddata

import matplotlib
matplotlib.use('Qt5Agg')
# from scroll import ScrollableWindow

class FittingSlaven:
    def __init__(self, file_location = None):
        self.results = [] # array of fitted data. Name is contained in outParams
        self.file_location = file_location
        self.para_dic = [] # loaded name of vectors
        self.dataFull = [] # all data loaded in numpy array 
        self.outParams = [] # name of fitted parameters
        self.initParam = [] # parameters used to initilize the fitting
        self.results_averaged = [] # results from fitting of averaged data
        self.threshold = 0

        if self.file_location == None:
            tkinter.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
            self.file_location = askopenfilename() # show an "Open" dialog box and return the path to the selected file
        
        self.loadFullData()

    def loadFullData(self):
        dirname = os.path.dirname(self.file_location)
        filename = os.path.splitext(os.path.basename(self.file_location))[0]
        json_file_location = filename + '.json'
        json_file_location = os.path.join(dirname, json_file_location)
        json_avr_file_location = filename + '_avr.json'
        json_avr_file_location = os.path.join(dirname, json_avr_file_location)
        with open(json_file_location, 'r', encoding='utf-8') as f:
            self.dataFull = json.load(f)
        with open(json_avr_file_location, 'r', encoding='utf-8') as f:
            self.averagedData = json.load(f)
    
    def fitData(self, plot=False):
        if len(self.initParam) == 0:
            self.setInitParam()

        self.setModelAndFit()
        self.saveData()
        plt.show()

    def plotData(self):
        tLast = self.averagedData[-1]['t']
        tfirst = self.averagedData[0]['t']
        total_measur = 0
        n_one = len(self.averagedData[0]['x'])


        y=[]
        z =[]
        for measur in self.averagedData:
            total_measur+=len(measur['y'])
            z.extend(measur['y'])
            y.extend(measur['x'])

        tvec = np.linspace(tfirst, tLast, total_measur//n_one)
        x = np.repeat(tvec, n_one)

        f_vec = np.linspace(min(y), max(y), n_one*2)
        xx, yy = np.meshgrid(tvec, f_vec)

        ##reshape
        xy = np.transpose(np.vstack((x,y)))
        z = np.array(z).reshape((-1,1))

        zz = griddata(xy, z, (xx, yy), method='linear')


        plt.figure()
        # plt.tricontourf(self.dataFull[1], self.dataFull[0], self.dataFull[2])
        plt.pcolormesh(xx,yy,np.log10(zz[:,:,0]))
        # plt.tripcolor(x, y, np.log10(z))
        plt.xlabel('temperature [K]')
        plt.ylabel('frequency [Hz]')
        self.saveFigure('orig', 'png')

        #log image
        plt.figure()
        plt.xscale('log')
        plt.pcolormesh(xx,yy,np.log10(zz[:,:,0]))
        plt.xlabel('temperature [K]')
        plt.ylabel('frequency [Hz]')
        self.saveFigure('orig_log', 'png')

    def saveData(self):
        if len(self.results):
            dirname = os.path.dirname(self.file_location)
            filename = os.path.splitext(os.path.basename(self.file_location))[0]
            fitted_file_location = filename + '_fitted.txt'
            fitted_file_location = os.path.join(dirname, fitted_file_location)
            header = '\t'.join(self.outParams)
            np.savetxt(fitted_file_location, self.results, delimiter = '\t', header = header)

        if len(self.results_averaged):
            dirname = os.path.dirname(self.file_location)
            filename = os.path.splitext(os.path.basename(self.file_location))[0]
            fitted_file_location = filename + '_fittedAveraged.txt'
            fitted_file_location = os.path.join(dirname, fitted_file_location)
            header = '\t'.join(self.outParams)
            np.savetxt(fitted_file_location, self.results_averaged, delimiter = '\t', header = header)
    
    def saveFigure(self, name, extension = 'pdf'):
        dirname = os.path.dirname(self.file_location)
        filename = os.path.splitext(os.path.basename(self.file_location))[0]
        image_file_name = filename+'_' + name + '.' + extension
        image_file_name = os.path.join(dirname,image_file_name)
        plt.savefig(image_file_name, bbox_inches='tight')

    # to implement
    def setInitParam(self):
        pass
    def setModelAndFit(self):
        pass


class SimpleFit(FittingSlaven):

    def setInitParam(self):
        sigma = (self.dataFull[0,0,-1] - self.dataFull[0,0,0])/10/2
        initParamDict = {
            'bg' : np.median(self.dataFull[0,2,:]),
            'center': self.dataFull[0,0, len(self.dataFull[0,0])//2],
            'sigma': sigma,
            'amplitude': np.pi*sigma*np.max(self.dataFull[0,2]),
        }

        self.initParam.append(initParamDict)

    def setModelAndFit(self, plot = False):

        # important to do the outParam, it is used later in the function
        self.outParams=['bg', 'freq', 'freqErr', 'fwhm', 'fwhmErr', 'amplitude', 'amplitudeErr', 'driving']
        background = ConstantModel(prefix='bg_')
        pars = background.make_params()

        peak = LorentzianModel(prefix='l1_')
        pars.update(peak.make_params())
        if plot:
            plt.figure()

        for i, data in enumerate(self.dataFull):
            #load init param
            pars['bg_c'].set(value = self.initParam[0]['bg'])
            pars['l1_center'].set(value=self.initParam[0]['center'])
            pars['l1_sigma'].set(value=self.initParam[0]['sigma'])
            pars['l1_amplitude'].set(value=self.initParam[0]['amplitude'])

            model = peak + background
            out = model.fit(data[2], pars, x=data[0])

            #save fit param as init param for the next
            self.initParam[0]['bg'] = out.best_values['bg_c']
            self.initParam[0]['center'] = out.best_values['l1_center']
            self.initParam[0]['sigma'] = out.best_values['l1_sigma']
            self.initParam[0]['amplitude'] = out.best_values['l1_amplitude']


            #important to have the same shape as the out
            self.results.append([out.best_values['bg_c'], 
                out.params['l1_center'].value,       out.params['l1_center'].stderr,
                out.params['l1_fwhm'].value,   out.params['l1_fwhm'].stderr,
                out.params['l1_height'].value, out.params['l1_height'].stderr,
                data[1][0]
                ])
            

            #for ploting the results
            if plot:
                columns = 15
                rows = len(self.dataFull)//columns
                if len(self.dataFull) % columns:
                    rows += 1
                
                plt.subplot(rows,columns,i+1)
                plt.plot(data[0], data[2], 'b')
                plt.xticks([]), plt.yticks([])
                plt.plot(data[0], out.init_fit, 'k--', label='initial fit')
                plt.plot(data[0], out.best_fit, 'r-', label='best fit')
                # plt.legend(loc='best')

        if plot:
            plt.subplots_adjust(hspace=0,left=0,bottom=0,right=1,top=1,wspace=0)
            plt.show()


class TwoPeaksFit(FittingSlaven):
    def setModelAndFit(self, plotData = False):
        # important to do the outParam, it is used later in the function
        self.outParams=['bg', 'temp', 'freq', 'freqErr', 'fwhm', 'fwhmErr', 'amplitude', 'amplitudeErr']
        self.outParams += ['freq2', 'freqErr2', 'fwhm2', 'fwhmErr2', 'amplitude2', 'amplitudeErr2']
        
        if plotData:
            plt.figure(figsize=(13,200))

        prev_temp = -1
        init_iterator = iter(self.init_params_per_temp)

        total_number = 0
        for elem in self.dataFull:
            total_number+=len(elem['y'])
        [elem.update(y=[ [ num*self.cf for num in mes ] for mes in elem['y'] ]) for elem in self.dataFull]

        if self.dataFull[0]['t'] != self.averagedData[0]['t']:
            self.dataFull = list(reversed(self.dataFull))

        curr_plot=1

        for _, measur in enumerate(self.dataFull):
            for _, y in enumerate(measur['y']):

                #set values from dictionary
                curr_temp = measur['t']
                if curr_temp!= prev_temp:
                    prev_temp = curr_temp
                    init_values_dict = next(init_iterator, self.init_params_per_temp[-1])

                background = ConstantModel(prefix='bg_')
                pars = background.make_params()
                pars['bg_c'].set(value = init_values_dict['bg'], min=0)

                
                if init_values_dict['center']:
                    peak = LorentzianModel(prefix='l1_')
                    pars.update(peak.make_params())
                    pars['l1_center'].set(value=init_values_dict['center'], min=0)
                    pars['l1_sigma'].set(value=init_values_dict['sigma'], min=0)
                    pars['l1_amplitude'].set(value=init_values_dict['amplitude'], min=0)
                
                if init_values_dict['center2']:
                    peak2 = LorentzianModel(prefix='l2_')
                    pars.update(peak2.make_params())
                    pars['l2_sigma'].set(value=init_values_dict['sigma2'], min=0)
                    pars['l2_amplitude'].set(value=init_values_dict['amplitude2'], min=0)

                    if init_values_dict['center']:
                        pars.add(name = 'split', min=0, max=100, vary=True)
                        pars['split'].set(value=init_values_dict['split'], min=0, max=100)
                        pars['l2_center'].set(value=init_values_dict['center2'], expr='l1_center+split', min=0)
                    else:
                        pars['l2_center'].set(value=init_values_dict['center2'], min=0)


                # set model
                if init_values_dict['center']:
                    if init_values_dict['center2']:
                        model = peak + peak2 + background
                    else:
                        model = peak + background
                else:
                    if init_values_dict['center2']:
                        model = peak2 + background

                out = model.fit(y, pars, x=measur['x'])

                for _, param in out.params.items():
                    if not param.stderr:
                        param.stderr = float('NaN')  
                
                self.results.append([out.best_values['bg_c']/self.cf,
                    measur['t'],
                    out.params.get('l1_center').value if 'l1_center' in out.params else None, out.params.get('l1_center').stderr if'l1_center'  in out.params else None,
                    out.params.get('l1_fwhm').value if 'l1_fwhm' in out.params else None,   out.params.get('l1_fwhm').stderr if'l1_fwhm'  in out.params else None,
                    out.params.get('l1_height').value/self.cf if 'l1_height' in out.params else None, out.params.get('l1_height').stderr/self.cf if'l1_height'  in out.params else None,
                    out.params.get('l2_center').value if 'l2_center' in out.params else None, out.params.get('l2_center').stderr if'l2_center'  in out.params else None,
                    out.params.get('l2_fwhm').value if 'l2_fwhm' in out.params else None,   out.params.get('l2_fwhm').stderr if'l2_fwhm'  in out.params else None,
                    out.params.get('l2_height').value/self.cf if 'l2_height' in out.params else None, out.params.get('l2_height').stderr/self.cf if'l2_height'  in out.params else None,
                    ])
                
                #for ploting the results
                if plotData:
                    columns = 7
                    rows = total_number//columns
                    if total_number % columns:
                        rows += 1
                    
                    plt.subplot(rows,columns,curr_plot)
                    curr_plot+=1
                    plt.plot(measur['x'], (y), 'b')
                    plt.yticks([])
                    plt.text(measur['x'][0], max(y), curr_plot, horizontalalignment='left', verticalalignment='top')
                    plt.xticks([])
                    plt.plot(measur['x'], (out.init_fit), 'k--', label='initial fit')
                    plt.plot(measur['x'], (out.best_fit), 'r-', label='best fit')
                    # plt.legend(loc='best')
                    # plt.show()

        if plotData:
            plt.subplots_adjust(hspace=0,left=0,bottom=0,right=1,top=1,wspace=0)
            self.saveFigure('allFitted')
            # ScrollableWindow(fig)

        self.results= np.array(self.results, dtype=np.float)
    
    def setInitParam(self, initDict=None, reverse = False, threshold=None, plotData = False, cf=1e14):
        self.cf = cf
        
        self.threshold = {'p1min': 0, 'p1max':len(self.averagedData),
                            'p2min': 0, 'p2max':len(self.averagedData)}
        if threshold:
            self.threshold.update(threshold)
        
        
        if plotData:
           plt.figure(figsize=(13,25))


        # multiply by factor to avoid errors in fitting
        [elem.update(y = [ num*self.cf for num in elem['y'] ]) for elem in self.averagedData]

        #this is result of this fitting that will be used for fitting the full data
        self.init_params_per_temp = []

        reverse = False if self.averagedData[0]['t'] > self.averagedData[-1]['t'] else True

        n = 0
        if reverse:
            n=-1
        #init dict for two peaks
        if initDict:
            initParamDict = initDict
            initParamDict['amplitude'] *= self.cf
            initParamDict['amplitude2'] *= self.cf
        else:
            sigma = (self.averagedData[n]['x'][-1] - self.averagedData[n]['x'][0])/10/2
            spl = 30
            initParamDict = {
                'bg' : np.median(self.averagedData[n]['y']),
                'center': self.averagedData[n]['x'][self.averagedData[n]['y'].index(max(self.averagedData[n]['y']))]-spl,
                'sigma': sigma,
                'amplitude': np.pi*sigma*np.max(self.averagedData[n]['y']),
                'center2': self.averagedData[n]['x'][self.averagedData[n]['y'].index(max(self.averagedData[n]['y']))],
                'sigma2': sigma,
                'amplitude2': np.pi*sigma*np.max(self.averagedData[n]['y']),
                'split': spl,
            }

        if reverse:
            self.averagedData = list(reversed(self.averagedData))

        for i, measurment in enumerate(self.averagedData):

            background = ConstantModel(prefix='bg_')
            pars = background.make_params()
            pars['bg_c'].set(value = initParamDict['bg'], min=0)

            if i>=self.threshold['p1min'] and i<=self.threshold['p1max']:
                peak = LorentzianModel(prefix='l1_')
                pars.update(peak.make_params())
                pars['l1_center'].set(value=initParamDict['center'], min=0)
                pars['l1_sigma'].set(value=initParamDict['sigma'], min=0)
                pars['l1_amplitude'].set(value=initParamDict['amplitude'], min=0)

            if i>=self.threshold['p2min'] and i<=self.threshold['p2max']:
                peak2 = LorentzianModel(prefix='l2_')
                pars.update(peak2.make_params())
                pars['l2_sigma'].set(value=initParamDict['sigma2'], min=0)
                pars['l2_amplitude'].set(value=initParamDict['amplitude2'], min=0)
                if 'l1_center' in pars:    
                    pars.add(name = 'split', value=initParamDict['split'], min=0, max=100, vary=True)
                    pars['l2_center'].set(value=initParamDict['center2'], expr='l1_center+split')
                else:
                    pars['l2_center'].set(value=initParamDict['center2'])

            if 'l1_center' in pars:
                if 'l2_center' in pars:
                    model = peak + peak2 + background
                else:
                    model = peak + background
            else:
                if 'l2_center' in pars:
                    model = peak2 + background
            
            out = model.fit(measurment['y'], pars, x=measurment['x'])
            
            #save data to file
            #save init params for the next one based on threshold
            self.update_init_params_and_save_data(initParamDict, out, measurment['t'], self.threshold, i)

            #for ploting the results
            if plotData:
                columns = 7
                rows = len(self.averagedData)//columns
                if len(self.averagedData) % columns:
                    rows += 1
                
                plt.subplot(rows,columns,i+1)
                plt.plot(measurment['x'], (measurment['y']), 'b')
                plt.yticks([])
                plt.xticks([])
                plt.plot(measurment['x'], (out.init_fit), 'k--', label='initial fit')
                plt.plot(measurment['x'], (out.best_fit), 'r-', label='best fit')
                plt.text(measurment['x'][0], max(measurment['y']), i+1, horizontalalignment='left', verticalalignment='top')
                # plt.legend(loc='best')
                # plt.show()

        if plotData:
            plt.subplots_adjust(hspace=0,left=0,bottom=0,right=1,top=1,wspace=0)
            self.saveFigure('averagedFitted')
            # ScrollableWindow(fig)

        self.results_averaged = np.array(self.results_averaged, dtype=np.float)

    #helper functions
    def update_params_values(self, pars, initParamDict, threshold, i):
        pars['l1_center'].set(value=initParamDict['center'])
        pars['l1_sigma'].set(value=initParamDict['sigma'], min=0)
        pars['l1_amplitude'].set(value=initParamDict['amplitude'], min=0)
        pars['bg_c'].set(value = initParamDict['bg'], min=0)

        if i<threshold:
            pars.add(name = 'split', value=initParamDict['split'], min=0, max=100, vary=True)
            pars['l2_center'].set(value=initParamDict['center2'], expr='l1_center+split')
            pars['l2_sigma'].set(value=initParamDict['sigma2'], min=0)
            pars['l2_amplitude'].set(value=initParamDict['amplitude2'], min=0)   
    
    def update_init_params_and_save_data(self, initParamDict, out, temp, threshold, i):
        for _, param in out.params.items():
            if not param.stderr:
                param.stderr = float('NaN')

        self.results_averaged.append([out.best_values['bg_c']/self.cf,
            temp,
            out.params.get('l1_center').value if 'l1_center' in out.params else None, out.params.get('l1_center').stderr if'l1_center'  in out.params else None,
            out.params.get('l1_fwhm').value if 'l1_fwhm' in out.params else None,   out.params.get('l1_fwhm').stderr if'l1_fwhm'  in out.params else None,
            out.params.get('l1_height').value/self.cf if 'l1_height' in out.params else None, out.params.get('l1_height').stderr/self.cf if'l1_height'  in out.params else None,
            out.params.get('l2_center').value if 'l2_center' in out.params else None, out.params.get('l2_center').stderr if'l2_center'  in out.params else None,
            out.params.get('l2_fwhm').value if 'l2_fwhm' in out.params else None,   out.params.get('l2_fwhm').stderr if'l2_fwhm'  in out.params else None,
            out.params.get('l2_height').value/self.cf if 'l2_height' in out.params else None, out.params.get('l2_height').stderr/self.cf if'l2_height'  in out.params else None,
            ])
        
        saveParamDict = initParamDict.copy()

        saveParamDict['bg'] = out.best_values.get('bg_c',0 )
        saveParamDict['center'] = out.best_values.get('l1_center',0 )
        saveParamDict['sigma'] = out.best_values.get('l1_sigma', 0 )
        saveParamDict['amplitude'] = out.best_values.get('l1_amplitude', 0 )
        saveParamDict['center2'] = out.best_values.get('l2_center', 0 )
        saveParamDict['sigma2'] = out.best_values.get('l2_sigma', 0 )
        saveParamDict['amplitude2'] = out.best_values.get('l2_amplitude', 0 )
        saveParamDict['split'] = out.params.get('split').value if 'split' in out.params else 0
        
        self.init_params_per_temp.append(saveParamDict)


        initParamDict['bg'] = out.best_values.get('bg_c', initParamDict['bg'] )
        initParamDict['center'] = out.best_values.get('l1_center', initParamDict['center'] )
        initParamDict['sigma'] = out.best_values.get('l1_sigma', initParamDict['sigma'] )
        initParamDict['amplitude'] = out.best_values.get('l1_amplitude', initParamDict['amplitude'] )
        initParamDict['center2'] = out.best_values.get('l2_center', initParamDict['center2'] )
        initParamDict['sigma2'] = out.best_values.get('l2_sigma', initParamDict['sigma2'] )
        initParamDict['amplitude2'] = out.best_values.get('l2_amplitude', initParamDict['amplitude2'] )
        initParamDict['split'] = out.params.get('split').value if 'split' in out.params else initParamDict['split']

# f2 = "V:/20200312/125553_spectrum_vs_freq_and_temperature/125553_spectrum_vs_freq_and_temperature.dat"
# f3 = "V:/20200314/113944_spectrum_vs_freq_and_temperature/113944_spectrum_vs_freq_and_temperature.dat"
# ff = 'Q:/20200310/104139_drivingAmpl_vs_Spectrum/104139_drivingAmpl_vs_Spectrum.dat'
# f4 = 'V:/20200403/114454_spectrum_vs_freq_and_temperature/114454_spectrum_vs_freq_and_temperature.dat'

th = {'p1min': 10, 'p2max':84}

# initDict = {
#                 'bg' : np.median(self.averagedData[n]['y']),
#                 'center': self.averagedData[n]['x'][self.averagedData[n]['y'].index(max(self.averagedData[n]['y']))]-spl,
#                 'sigma': sigma,
#                 'amplitude': np.pi*sigma*np.max(self.averagedData[n]['y']),
#                 'center2': self.averagedData[n]['x'][self.averagedData[n]['y'].index(max(self.averagedData[n]['y']))],
#                 'sigma2': sigma,
#                 'amplitude2': np.pi*sigma*np.max(self.averagedData[n]['y']),
#                 'split': spl,
#             }

fit = TwoPeaksFit()
# fit.plotData()
fit.setInitParam(initDict=None, reverse = None, threshold=th, plotData = True)
fit.setModelAndFit(plotData=True)
fit.saveData()