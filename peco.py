# Declaration: Peco is based on arXiv:1711.10925
# Author: Hongze Gu
# Date: 2023.9.3


#############
## Import  ##
#############
import matplotlib.pyplot as plt
import os, glob, re
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.optim
import torch.nn as nn
import torchvision.transforms as tvt
import torchvision

# modify add
def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

#from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# GUI
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter.messagebox import askokcancel

#############
## Seting  ##
#############
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
sigma = 25
sigma_ = sigma/255.







class denoiseEngine(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=15)
        self.pack(fill=BOTH, expand=YES)
        
        self.i = 0
        self.net_input = None

        # application variables
        self.styleOfWindow = ttk.StringVar(value = "solar")
        self.dataPath = ttk.StringVar(value = "./data/")
        self.totalIterations = ttk.IntVar(value = 2400)
        self.saveEvery = ttk.IntVar(value = 800)
        self.PLOT = ttk.BooleanVar(value = True)
        self.processDataFormat = ttk.StringVar(value = "tiff") # tiff or png 
        self.learningRate = ttk.DoubleVar(value = 0.01)
        self.optimizer = ttk.StringVar(value = "adam") # 'adam' or 'LBFGS'

        # header and labelframe option container
        optionText = "Complete the form to begin your denoising"
        self.optionLabelFrame = ttk.Labelframe(self, text=optionText, padding=15)
        self.optionLabelFrame.pack(fill=X, expand=YES, anchor=N)

        # create widgets
        self.createChangeStyleRow()
        self.createBlankRow()
        self.createDataFolderRow()
        self.createBlankRow()
        self.createIterationsNumberRow()
        self.createBlankRow()
        self.createProcessDataTypeRow()
        self.createBlankRow()
        self.createLeaningRateRow()
        self.createBlankRow()
        self.createOptimizerChooseRow()

        ttk.Sizegrip(master=self, bootstyle=PRIMARY).pack(fill=X, expand=YES)
        self.createDenoiseCheckButtonRow()
        self.progressbar = ttk.Progressbar(
            #master=self, 
            mode=INDETERMINATE, 
            bootstyle=(STRIPED, SUCCESS)
        )
        self.progressbar.pack(fill=X, expand=YES)
        self.progressbar.start(15)

        

    def createChangeStyleRow(self):
        dataStyleRow = ttk.Frame(self.optionLabelFrame)
        dataStyleRow.pack(fill=X, expand=YES)
        
        themeShow = ttk.Frame(dataStyleRow, padding=(10, 0, 10, 0)) # padding = (left, top, right, botom)
        themeShow.pack(fill=X, expand=YES)

        showThemeText = ttk.Label(themeShow, text="Choose style", width=11)
        showThemeText.pack(side=LEFT, padx=(5, 0))

        styleNames = ttk.Style().theme_names()
        themeCombobox = ttk.Combobox(themeShow, textvariable=self.styleOfWindow, values=styleNames)
        themeCombobox.current(15) # 13 --> solar 15 --> vapor
        themeCombobox.pack(side=LEFT, fill=X, expand=YES, padx=5)
        ttk.Style().theme_use(self.styleOfWindow.get())

        themeShowSelection = ttk.Label(
            master=themeShow,
            text="vapor",
            font="-size 24 -weight bold"
        )
        themeShowSelection.pack(side=LEFT, padx=(5,0))

        def changeTheme(event):
            ttk.Style().theme_use(themeCombobox.get())
            #print(themeCombobox.get())
            themeShowSelection.configure(text = themeCombobox.get())
            themeCombobox.select_clear()
        
        themeCombobox.bind('<<ComboboxSelected>>', changeTheme)
    

    def createBlankRow(self):
        dataBlankRow = ttk.Frame(self.optionLabelFrame)
        dataBlankRow.pack(fill=X, expand=YES)

        dataBlankLable = ttk.Label(dataBlankRow, text="", width=10)
        dataBlankLable.pack(side=LEFT)


    def createDataFolderRow(self):
        dataFolderRow = ttk.Frame(self.optionLabelFrame)
        dataFolderRow.pack(fill=X, expand=YES)

        dataFolderLable = ttk.Label(dataFolderRow, text="Data Folder", width=11)
        dataFolderLable.pack(side=LEFT, padx=(15, 0))

        dataFolderEntry = ttk.Entry(dataFolderRow, textvariable=self.dataPath)
        dataFolderEntry.pack(side=LEFT, fill=X, expand=YES, padx=5)

        dataFolderBrowse = ttk.Button(master=dataFolderRow, text="Browse", command=self.browseData, width=8)
        dataFolderBrowse.pack(side=LEFT, padx=5)



    def createIterationsNumberRow(self):
        inputRow = ttk.Frame(self.optionLabelFrame)
        inputRow.pack(fill=X, expand=YES)

        dataInputLable = ttk.Label(inputRow, text="Total iterations")
        dataInputLable.pack(side=LEFT, padx=(15, 0))

        dataInputEntry = ttk.Entry(inputRow, textvariable=self.totalIterations)
        dataInputEntry.pack(side=LEFT, fill=BOTH, expand=YES, padx=5)

        dataInputLable = ttk.Label(inputRow, text="Save(Plot) every")
        dataInputLable.pack(side=LEFT, padx=(15, 0))

        dataInputEntry = ttk.Entry(inputRow, textvariable=self.saveEvery)
        dataInputEntry.pack(side=LEFT, fill=BOTH, expand=YES, padx=5)


    def createProcessDataTypeRow(self):
        formatRow = ttk.Frame(self.optionLabelFrame)
        formatRow.pack(fill=X, expand=YES)
        formatLabel = ttk.Label(formatRow, text="Process data format", width=25)
        formatLabel.pack(side=LEFT, padx=(15, 0))

        format1 = ttk.Radiobutton(
            master=formatRow, 
            text="png", 
            variable=self.processDataFormat, 
            value="png"
        )
        format1.pack(side=LEFT)

        format2 = ttk.Radiobutton(
            master=formatRow, 
            text="tiff", 
            variable=self.processDataFormat, 
            value="tiff"
        )
        format2.pack(side=LEFT, padx=15)
        format2.invoke()

        def plotWhenDenoising():
            print(self.PLOT.get())
        
        showImageWhenDenoising = ttk.Checkbutton(
            master=formatRow,
            command=plotWhenDenoising,
            text="Plot process",
            onvalue=True, 
            offvalue=False,
            variable=self.PLOT
        )
        showImageWhenDenoising.pack(side=LEFT)



    def createLeaningRateRow(self):
        inputRow = ttk.Frame(self.optionLabelFrame)
        inputRow.pack(fill=X, expand=YES)

        dataInputLable = ttk.Label(inputRow, text="Learning Rate")
        dataInputLable.pack(side=LEFT, padx=(15, 0))

        dataInputEntry = ttk.Entry(inputRow, textvariable=self.learningRate)
        dataInputEntry.pack(side=LEFT, fill=BOTH, expand=YES, padx=5)



    def createOptimizerChooseRow(self):
        formatRow = ttk.Frame(self.optionLabelFrame)
        formatRow.pack(fill=X, expand=YES)
        formatLabel = ttk.Label(formatRow, text="Optimizer choice", width=25)
        formatLabel.pack(side=LEFT, padx=(15, 0))

        format1 = ttk.Radiobutton(
            master=formatRow, 
            text="ADAM", 
            variable=self.optimizer, 
            value="adam"
        )
        format1.pack(side=LEFT)

        format2 = ttk.Radiobutton(
            master=formatRow, 
            text="LBFGS", 
            variable=self.optimizer, 
            value="LBFGS"
        )
        format2.pack(side=LEFT, padx=15)
        format1.invoke()




    def createDenoiseCheckButtonRow(self):
        ttk.Separator(self, bootstyle=DARK).pack(expand=YES, fill=X, pady=10)
        
        denoiseButton = ttk.Button(self, text="Denoise data", command=self.callDenoiseData, width=25, bootstyle=SUCCESS)
        denoiseButton.pack(side=LEFT, padx=10, pady=10, expand=YES, fill=X)

        checkButton = ttk.Button(self, text="Check results", command=self.checkResults, width=25, bootstyle=INFO)
        checkButton.pack(side=LEFT, padx=10, pady=10, expand=YES, fill=X)

        cleanButton = ttk.Button(self, text="Clean up", command=self.cleanUp, width=25, bootstyle=WARNING)
        cleanButton.pack(side=LEFT, padx=10, pady=10, expand=YES, fill=X)

        clearButton = ttk.Button(self, text="Clear all", command=self.clearAll, width=25, bootstyle=DANGER)
        clearButton.pack(side=LEFT, padx=10, pady=10, expand=YES, fill=X)


    def callDenoiseData(self):
        self.progressbar.stop()
        self.progressbar.destroy()
        self.progressbar=ttk.Progressbar(
            # master=self, 
            mode=INDETERMINATE, 
            bootstyle=(STRIPED, SUCCESS)
        )
        self.progressbar.pack(fill=X, expand=YES)
        self.progressbar.start(20)
        
        self.denoiseData()
        print("Done!")

        
    def checkResults(self):
        self.progressbar.stop()
        self.progressbar.destroy()
        self.progressbar=ttk.Progressbar(
            #master=self, 
            mode=INDETERMINATE, 
            bootstyle=(STRIPED, INFO)
        )
        self.progressbar.pack(fill=X, expand=YES)
        self.progressbar.start(30)
        
       
        itersFileList = self.getPathList(self.dataPath.get(), "txt")
        originFileList = self.getPathList(self.dataPath.get(), "data")
        
        howMany = len(originFileList)
        iters = len(itersFileList)
        saveNumber = iters//howMany
        flag = iters%howMany
        if flag != 0:
            print(f"data files number can't be divided by txt files. data % txt = {flag}")

        for i in range(howMany):
            sampleName = originFileList[i]
            originFile = np.loadtxt(sampleName)
            for j in range(saveNumber):
                iterName = itersFileList[i*saveNumber+j]
                iterFile = np.loadtxt(iterName)
                print(f"width = {iterFile.shape[-1]}, height = {iterFile.shape[0]}" )
                noiseFile = originFile - iterFile
                plt.subplot(131)
                plt.imshow(originFile,'terrain_r')
                plt.colorbar()
                plt.title(sampleName)

                plt.subplot(132)
                plt.imshow(iterFile,'terrain_r')
                plt.colorbar()
                plt.title(iterName)

                plt.subplot(133)
                plt.imshow(noiseFile,'terrain_r')
                plt.colorbar()
                plt.title("Noise")

                plt.show()

        print("Done!")


    def cleanUp(self):
        self.progressbar.stop()
        self.progressbar.destroy()
        self.progressbar=ttk.Progressbar(
            #master=self, 
            mode=INDETERMINATE, 
            bootstyle=(STRIPED, WARNING)
        )
        self.progressbar.pack(fill=X, expand=YES)
        self.progressbar.start(30)
        
        areYouSure = askokcancel(title="Clear up warning!", message="This operation will disable check results button!")
        if areYouSure:
            dataFolderRemains = self.getPathList(self.dataPath.get(), "info")
            for remain in dataFolderRemains:
                os.remove(remain)
            dataFolderRemains = self.getPathList(self.dataPath.get(), "data")
            for remain in dataFolderRemains:
                os.remove(remain)
            resultsFolderRemains = self.getPathList(self.dataPath.get(), "txt")
            for remain in resultsFolderRemains:
                os.remove(remain)
            
            resultsFolderRemains = self.getPathList(self.dataPath.get(), self.processDataFormat.get())
            for remain in resultsFolderRemains:
                os.remove(remain)
        
        print("Done!")


    def clearAll(self):
        self.progressbar.stop()
        self.progressbar.destroy()
        self.progressbar=ttk.Progressbar(
            #master=self, 
            mode=INDETERMINATE, 
            bootstyle=(STRIPED, DANGER)
        )
        self.progressbar.pack(fill=X, expand=YES)
        self.progressbar.start(40)
        
        areYouSure = askokcancel(title="Clear all warning!", message="Are you sure to delete all the files in the data folder?")
        if areYouSure:
            for file in glob.glob(self.dataPath.get()+"*"):
                os.remove(file)
        print("Done!")


    def browseData(self):
        path = askdirectory(initialdir=self.dataPath.get(), title="Browse Directory")
        if path:
            self.dataPath.set(path+"/")



    # Data preparation
    def getPathList(self, path, fileType):
        #Return all the images path end with ".fileType" in the folder indicated by path.
        return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.'+fileType)]


    # Data preparation
    def analyseITXfile(self, path):
        # load the itx file in pathList and process them.
        pathList = self.getPathList(path, 'itx')
        
        def getCriticalData(path):
            readfile = open(path)
            contentOfFile = readfile.read()
            readfile.close()

            #replace all the NAN to 0
            contentOfFile = re.sub('NAN', '0', contentOfFile, flags=re.I)
            
            dataStart=re.search("BEGIN\n+",contentOfFile,re.I).span()[1]
            dataStop=re.search("\n+END",contentOfFile,re.I).span()[0]
            infoWithoutData = re.sub(contentOfFile[dataStart:dataStop],"", contentOfFile)
            outputFile1 = open(path.split(".itx")[-2] + ".info", "w")
            outputFile1.write(infoWithoutData)
            outputFile1.close()

            '''
            deleteLineWithNAN = re.compile(r'\n.*NAN.*\n', re.I)
            while True:
                linesWithNAN = deleteLineWithNAN.findall(contentOfFile)
                if len(linesWithNAN)==0: #Notice that if "\nNAN\nNAN\nNAN\nNAN\n", then need 3 times to clear them up.
                    break
                for line in linesWithNAN:
                    contentOfFile = re.sub(line, "\n", contentOfFile)
            '''
            
            #Cut head and tail information after re.sub can make sure that it works.
            dataStart=re.search("BEGIN\n+",contentOfFile,re.I).span()[1]
            dataStop=re.search("\n+END",contentOfFile,re.I).span()[0]
            data=contentOfFile[dataStart:dataStop]
            
            #Cut the \t in the front of each line.
            deleteFrontTab = re.compile("^\t", re.M)
            data = deleteFrontTab.sub("", data)

            outputFile2 = open(path.split(".itx")[-2]+".data", "w")
            outputFile2.write(data)
            outputFile2.close()
        
        processBar = tqdm(pathList)
        for p in processBar:
            processBar.set_description(".itx --> .data")
            getCriticalData(p)


    # Data preparation
    def convert_ToIMAGE_thenSave(self, fpath, mode='png'):
        paths = self.getPathList(fpath, 'data')
        weights = []
        shapes = []
        for path in paths:
            data = np.loadtxt(path, dtype=np.float32)
            data[data<0] = 0

            shapes.append(data.shape)
            weights.append(data.sum())
            if mode == 'png':
                scale = data.max()/255.0
                data = (data/scale).astype(np.uint8)
                plt.imsave(path.split(".data")[-2]+'.png', data, cmap='gray')
            elif mode == 'tiff':
                scale = data.max()/255.0
                data = data/scale
                image = Image.fromarray(data)
                #image.show()
                image.save(path.split(".data")[-2]+'.tiff')

        return self.getPathList(fpath, mode), shapes, weights



    def resize_image(self, img, d=32):
        '''Make dimensions divisible by `d`'''

        new_size = (img.size[0] - img.size[0] % d, 
                    img.size[1] - img.size[1] % d)

        img_resized = img.resize(new_size, Image.LANCZOS) # Lanczos
        return img_resized



    def pil_to_np(self, img_PIL):
        '''Converts image in PIL format to np.array.
        
        From W x H x C [0...255] to C x W x H [0..1]
        '''
        ar = np.array(img_PIL)

        if len(ar.shape) == 3:
            ar = ar.transpose(2,0,1)
        else:
            ar = ar[None, ...]

        return ar.astype(np.float32) / 255.


    class Concat(nn.Module):
        def __init__(self, dim, *args):
            super().__init__()
            self.dim = dim

            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        def forward(self, input):
            inputs = []
            for module in self._modules.values():
                inputs.append(module(input))

            inputs_shapes2 = [x.shape[2] for x in inputs]
            inputs_shapes3 = [x.shape[3] for x in inputs]        

            if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
                inputs_ = inputs
            else:
                target_shape2 = min(inputs_shapes2)
                target_shape3 = min(inputs_shapes3)

                inputs_ = []
                for inp in inputs: 
                    diff2 = (inp.size(2) - target_shape2) // 2 
                    diff3 = (inp.size(3) - target_shape3) // 2 
                    inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

            return torch.cat(inputs_, dim=self.dim)

        def __len__(self):
            return len(self._modules)



    def bn(self, num_features):
        return nn.BatchNorm2d(num_features)



    class Downsampler(nn.Module):
        '''
            http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
        '''
        def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
            super().__init__()
            
            assert phase in [0, 0.5], 'phase should be 0 or 0.5'

            if kernel_type == 'lanczos2':
                support = 2
                kernel_width = 4 * factor + 1
                kernel_type_ = 'lanczos'

            elif kernel_type == 'lanczos3':
                support = 3
                kernel_width = 6 * factor + 1
                kernel_type_ = 'lanczos'

            elif kernel_type == 'gauss12':
                kernel_width = 7
                sigma = 1/2
                kernel_type_ = 'gauss'

            elif kernel_type == 'gauss1sq2':
                kernel_width = 9
                sigma = 1./np.sqrt(2)
                kernel_type_ = 'gauss'

            elif kernel_type in ['lanczos', 'gauss', 'box']:
                kernel_type_ = kernel_type

            else:
                assert False, 'wrong name kernel'
                
                
            # note that `kernel width` will be different to actual size for phase = 1/2
            self.kernel = self.get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
            
            downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
            downsampler.weight.data[:] = 0
            downsampler.bias.data[:] = 0

            kernel_torch = torch.from_numpy(self.kernel)
            for i in range(n_planes):
                downsampler.weight.data[i, i] = kernel_torch       

            self.downsampler_ = downsampler

            if preserve_size:

                if  self.kernel.shape[0] % 2 == 1: 
                    pad = int((self.kernel.shape[0] - 1) / 2.)
                else:
                    pad = int((self.kernel.shape[0] - factor) / 2.)
                    
                self.padding = nn.ReplicationPad2d(pad)
            
            self.preserve_size = preserve_size
            
        def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
            assert kernel_type in ['lanczos', 'gauss', 'box']
            
            # factor  = float(factor)
            if phase == 0.5 and kernel_type != 'box': 
                kernel = np.zeros([kernel_width - 1, kernel_width - 1])
            else:
                kernel = np.zeros([kernel_width, kernel_width])
            
                
            if kernel_type == 'box':
                assert phase == 0.5, 'Box filter is always half-phased'
                kernel[:] = 1./(kernel_width * kernel_width)
                
            elif kernel_type == 'gauss': 
                assert sigma, 'sigma is not specified'
                assert phase != 0.5, 'phase 1/2 for gauss not implemented'
                
                center = (kernel_width + 1.)/2.
                print(center, kernel_width)
                sigma_sq =  sigma * sigma
                
                for i in range(1, kernel.shape[0] + 1):
                    for j in range(1, kernel.shape[1] + 1):
                        di = (i - center)/2.
                        dj = (j - center)/2.
                        kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
                        kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)
            elif kernel_type == 'lanczos': 
                assert support, 'support is not specified'
                center = (kernel_width + 1) / 2.

                for i in range(1, kernel.shape[0] + 1):
                    for j in range(1, kernel.shape[1] + 1):
                        
                        if phase == 0.5:
                            di = abs(i + 0.5 - center) / factor  
                            dj = abs(j + 0.5 - center) / factor 
                        else:
                            di = abs(i - center) / factor
                            dj = abs(j - center) / factor
                        
                        
                        pi_sq = np.pi * np.pi

                        val = 1
                        if di != 0:
                            val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                            val = val / (np.pi * np.pi * di * di)
                        
                        if dj != 0:
                            val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                            val = val / (np.pi * np.pi * dj * dj)
                        
                        kernel[i - 1][j - 1] = val
                    
                
            else:
                assert False, 'wrong method name'
            
            kernel /= kernel.sum()
            
            return kernel
        
        def forward(self, input):
            if self.preserve_size:
                x = self.padding(input)
            else:
                x= input
            self.x = x
            return self.downsampler_(x)



    def conv(self, in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
        downsampler = None
        if stride != 1 and downsample_mode != 'stride':

            if downsample_mode == 'avg':
                downsampler = nn.AvgPool2d(stride, stride)
            elif downsample_mode == 'max':
                downsampler = nn.MaxPool2d(stride, stride)
            elif downsample_mode  in ['lanczos2', 'lanczos3']:
                downsampler = self.Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
            else:
                assert False

            stride = 1

        padder = None
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            padder = nn.ReflectionPad2d(to_pad)
            to_pad = 0
    
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


        layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
        return nn.Sequential(*layers)



    class Swish(nn.Module):
        """
            https://arxiv.org/abs/1710.05941
            The hype was so huge that I could not help but try it
        """
        def __init__(self):
            super().__init__()
            self.s = nn.Sigmoid()

        def forward(self, x):
            return x * self.s(x)




    def act(self, act_fun = 'LeakyReLU'):
        '''
            Either string defining an activation function or module (e.g. nn.ReLU)
        '''
        if isinstance(act_fun, str):
            if act_fun == 'LeakyReLU':
                return nn.LeakyReLU(0.2, inplace=True)
            elif act_fun == 'Swish':
                return self.Swish()
            elif act_fun == 'ELU':
                return nn.ELU()
            elif act_fun == 'none':
                return nn.Sequential()
            else:
                assert False
        else:
            return act_fun()



    def skip(self,
            num_input_channels=2, num_output_channels=3, 
            num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
            filter_size_down=3, filter_size_up=3, filter_skip_size=1,
            need_sigmoid=True, need_bias=True, 
            pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
            need1x1_up=True):
        """Assembles encoder-decoder with skip connections.

        Arguments:
            act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
            pad (string): zero|reflection (default: 'zero')
            upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
            downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

        """
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down) 

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
            upsample_mode   = [upsample_mode]*n_scales

        if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
            downsample_mode   = [downsample_mode]*n_scales
        
        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
            filter_size_down   = [filter_size_down]*n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
            filter_size_up   = [filter_size_up]*n_scales

        last_scale = n_scales - 1 

        cur_depth = None

        model = nn.Sequential()
        model_tmp = model

        input_depth = num_input_channels
        for i in range(len(num_channels_down)):

            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add(self.Concat(1, skip, deeper))
            else:
                model_tmp.add(deeper)
            
            model_tmp.add(self.bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                skip.add(self.conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                skip.add(self.bn(num_channels_skip[i]))
                skip.add(self.act(act_fun))
                
            # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

            deeper.add(self.conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
            deeper.add(self.bn(num_channels_down[i]))
            deeper.add(self.act(act_fun))

            deeper.add(self.conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
            deeper.add(self.bn(num_channels_down[i]))
            deeper.add(self.act(act_fun))

            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                # The deepest
                k = num_channels_down[i]
            else:
                deeper.add(deeper_main)
                k = num_channels_up[i + 1]

            deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

            model_tmp.add(self.conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(self.bn(num_channels_up[i]))
            model_tmp.add(self.act(act_fun))


            if need1x1_up:
                model_tmp.add(self.conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                model_tmp.add(self.bn(num_channels_up[i]))
                model_tmp.add(self.act(act_fun))

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        model.add(self.conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            model.add(nn.Sigmoid())

        return model



    def pil_to_np(self, img_PIL):
        '''Converts image in PIL format to np.array.
        
        From W x H x C [0...255] to C x W x H [0..1]
        '''
        ar = np.array(img_PIL)

        if len(ar.shape) == 3:
            ar = ar.transpose(2,0,1)
        else:
            ar = ar[None, ...]

        return ar.astype(np.float32) / 255.



    def get_image(self, path, imsize=-1):
        """Load an image and resize to a cpecific size. 

        Args: 
            path: path to image
            imsize: tuple or scalar with dimensions; -1 for `no resize`
        """
        def load(path):
            """Load PIL image."""
            img = Image.open(path)
            if path.endswith('.png'): img=img.convert('RGB')
            return img
        

        img = load(path)

        if isinstance(imsize, int):
            imsize = (imsize, imsize)

        if imsize[0]!= -1 and img.size != imsize:
            if imsize[0] > img.size[0]:
                img = img.resize(imsize, Image.BICUBIC)
            else:
                img = img.resize(imsize, Image.ANTIALIAS)

        img_np = self.pil_to_np(img)

        return img, img_np



    def plot_image_grid(self, images_np, nrow =8, factor=1, interpolation='lanczos'):
        """Draws images in a grid
        
        Args:
            images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
            nrow: how many images will be in one row
            factor: size if the plt.figure 
            interpolation: interpolation used in plt.imshow
        """
        def get_image_grid(images_np, nrow=8):
            '''Creates a grid from a list of images by concatenating them.'''
            images_torch = [torch.from_numpy(x) for x in images_np]
            torch_grid = torchvision.utils.make_grid(images_torch, nrow)
            
            return torch_grid.numpy()


        n_channels = max(x.shape[0] for x in images_np)
        assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
        
        images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

        grid = get_image_grid(images_np, nrow)
        
        plt.figure()#figsize=(len(images_np) + factor, 12 + factor))
        
        if images_np[0].shape[0] == 1:
            plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
        else:
            plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
        
        plt.show()
        
        return grid



    def np_to_torch(self, img_np):
        '''Converts image in numpy.array to torch.Tensor.

        From C x W x H [0..1] to  C x W x H [0..1]
        '''
        return torch.from_numpy(img_np)[None, :]
    


    def torch_to_np(self, img_var):
        '''Converts an image in torch.Tensor format to np.array.

        From 1 x C x W x H [0..1] to  C x W x H [0..1]
        '''
        return img_var.detach().cpu().numpy()[0]



    def get_noise(self, input_depth, method, spatial_size, noise_type='u', var=1./10):
        """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
        initialized in a specific way.
        Args:
            input_depth: number of channels in the tensor
            method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
            spatial_size: spatial size of the tensor to initialize
            noise_type: 'u' for uniform; 'n' for normal
            var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
        """
        def fill_noise(x, noise_type):
            """Fills tensor `x` with noise of type `noise_type`."""
            if noise_type == 'u':
                x.uniform_()
            elif noise_type == 'n':
                x.normal_() 
            else:
                assert False
        

        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        if method == 'noise':
            shape = [1, input_depth, spatial_size[0], spatial_size[1]]
            net_input = torch.zeros(shape)
            
            fill_noise(net_input, noise_type)
            net_input *= var            
        elif method == 'meshgrid': 
            assert input_depth == 2
            X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
            meshgrid = np.concatenate([X[None,:], Y[None,:]])
            net_input=  self.np_to_torch(meshgrid)
        else:
            assert False
            
        return net_input



    def Regenerate_toITX(self, infoPath, txtPath, iterations):
        #pathsOfINFO = getPathlist(fpath, "info")
        #pathsOfTXT = getPathlist(fpath, "txt")
        #numOfFiles = len(pathsOfTXT)

        #for i in tqdm(range(numOfFiles), "Regenerating data"):
        #for i in range(numOfFiles):
        readINFO = open(infoPath)
        contentOfINFO = readINFO.read()
        readINFO.close()

        readTXT = open(txtPath)
        contentOfTXT = readTXT.read()
        readTXT.close()
        
        theStringOfShapeInfo = re.compile("[(](.*)[)]") # Match the string between "(" and ")"
        dataShape = "(" + re.findall(theStringOfShapeInfo, contentOfINFO)[0] + ")"
        
        #contentOfINFO = contentOfINFO.replace(dataShape, str(dataSetShape[i])) # The cuts of NAN may change the shape of data. Update the new shape and replace the ".info".
        
        nameStart = re.search(dataShape, contentOfINFO, re.I).span()[1] + 1 #use +1 to cancel the ")"
        nameStop = re.search("BEGIN\n+", contentOfINFO, re.I).span()[0]
        waitToBeReplace = contentOfINFO[nameStart:nameStop].strip()
        #print("start")
        #print(waitToBeReplace)
        #print("end")
        contentOfINFO = contentOfINFO.replace(waitToBeReplace, waitToBeReplace+f"_{iterations}iteration_deno")

        dataStart = re.search("BEGIN\n+", contentOfINFO, re.I).span()[1]
        dataStop = re.search("\n+END", contentOfINFO, re.I).span()[0]
        #"\n+" means \n 1~∞

        outputFile = open(txtPath.split(".txt")[-2]+".itx", "w")
        outputFile.write(contentOfINFO[0:dataStart])
        outputFile.write(contentOfTXT)
        outputFile.write(contentOfINFO[dataStop:])
        #print(contentOfINFO[dataStop:])

        outputFile.close()



    def get_params(self, opt_over, net, net_input, downsampler=None):
        '''Returns parameters that we want to optimize over.

        Args:
            opt_over: comma separated list, e.g. "net,input" or "net"
            net: network
            net_input: torch.Tensor that stores input `z`
        '''
        opt_over_list = opt_over.split(',')
        params = []
        
        for opt in opt_over_list:
        
            if opt == 'net':
                params += [x for x in net.parameters() ]
            elif  opt=='down':
                assert downsampler is not None
                params = [x for x in downsampler.parameters()]
            elif opt == 'input':
                net_input.requires_grad = True
                params += [net_input]
            else:
                assert False, 'what is it?'
                
        return params



    def optimize(self, optimizer_type, parameters, closure, LR, num_iter):
        """Runs optimization loop.

        Args:
            optimizer_type: 'LBFGS' of 'adam'
            parameters: list of Tensors to optimize over
            closure: function, that returns loss variable
            LR: learning rate
            num_iter: number of iterations 
        """
        if optimizer_type == 'LBFGS':
            # Do several steps with adam first
            optimizer = torch.optim.Adam(parameters, lr=0.001)
            for j in range(100):
                optimizer.zero_grad()
                closure()
                optimizer.step()

            print('Starting optimization with LBFGS')        
            def closure2():
                optimizer.zero_grad()
                return closure()
            optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
            optimizer.step(closure2)

        elif optimizer_type == 'adam':
            print('Starting optimization with ADAM')
            optimizer = torch.optim.Adam(parameters, lr=LR)
            
            for j in range(num_iter):
                optimizer.zero_grad()
                closure()
                optimizer.step()
        else:
            assert False



    def denoiseData(self):
        ###############
        ## Load itx  ##
        ###############
        fpath = self.dataPath.get()
        self.analyseITXfile(fpath)
        fnameSet, shapesOrigin, weightsOrigin = self.convert_ToIMAGE_thenSave(fpath, self.processDataFormat.get())

        for currentImage, fname in enumerate(fnameSet):
            #################
            ## load image  ##
            #################
            img_noisy_pil = self.resize_image(self.get_image(fname, imsize)[0], d=32)
            img_noisy_np = self.pil_to_np(img_noisy_pil)
                
            # As we don't have ground truth
            img_pil = img_noisy_pil
            img_np = img_noisy_np
                
            if self.PLOT.get():
                self.plot_image_grid([img_np], 4, 5)

            ############
            ## Setup  ##
            ############
            imageSaveType = self.processDataFormat.get() # 'tiff' or 'png'
            PLOT = self.PLOT.get()
            INPUT = 'noise' # 'meshgrid'
            pad = 'reflection'
            OPT_OVER = 'net' # 'net,input'

            reg_noise_std = 1./30. # set to 1./20. for sigma=50
            LR = self.learningRate.get()

            OPTIMIZER = self.optimizer.get() # 'adam' or 'LBFGS'
            show_every = self.saveEvery.get()
            exp_weight = 0.99


            num_iter = self.totalIterations.get()
            input_depth = 3 # Write channels of your image
            if imageSaveType == 'tiff':
                input_depth = 1
            #elif imageSaveType == 'png':
            #    input_depth = 3
            figsize = 5 

            net = self.skip(
                        input_depth, input_depth, 
                        num_channels_down = [8, 16, 32, 64, 128], 
                        num_channels_up   = [8, 16, 32, 64, 128],
                        num_channels_skip = [0, 0, 0, 4, 4], 
                        upsample_mode='bilinear',
                        need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

            net = net.type(dtype)

            self.net_input = self.get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

            # Compute number of parameters
            s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
            print ('Number of params: %d' % s)

            # Loss
            mse = torch.nn.MSELoss().type(dtype)

            img_noisy_torch = self.np_to_torch(img_noisy_np).type(dtype)



            ###############
            ## Optimize  ##
            ###############
            net_input_saved = self.net_input.detach().clone()
            noise = self.net_input.detach().clone()

            self.i = 0
            def closure():       
                if reg_noise_std > 0:
                    self.net_input = net_input_saved + (noise.normal_() * reg_noise_std)
                
                out = net(self.net_input)
                
                # Smoothing
                '''
                if out_avg is None:
                    out_avg = out.detach()
                else:
                    out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                '''        
                total_loss = mse(out, img_noisy_torch)
                total_loss.backward()
                    
                
                #psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
                #psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
                #psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
                
                # Note that we do not have GT for the ARPES data.
                # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
                #print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
                print ('Iteration %05d    Loss %f' % (self.i, total_loss.item()), '\r', end='')
                
                if self.i % show_every == 0:
                    shape = shapesOrigin[currentImage]
                    out_resize = tvt.Resize(shape, antialias=True)(out)
                    out_np = self.torch_to_np(out_resize)
                    
                    out_np = out_np.mean(axis=0)
                    weight = weightsOrigin[currentImage]
                    out_np *= weight/out_np.sum()
                    
                    infoPath = fname.split("."+imageSaveType)[-2] + ".info"
                    txtPath = fname.split("."+imageSaveType)[-2] + f"_{self.i}iteration_deno.txt"
                    np.savetxt(txtPath, out_np, delimiter="\t") # 3 channels --> 1 channel
                    self.Regenerate_toITX(infoPath, txtPath, self.i)

                    if  PLOT:
                        out_np = self.torch_to_np(out)
                        self.plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
                    
                
                self.i += 1

                return total_loss

            p = self.get_params(OPT_OVER, net, self.net_input)
            self.optimize(OPTIMIZER, p, closure, LR, num_iter)


            #################
            ## Final show  ##
            #################
            out = net(self.net_input)
            out_tmp = self.torch_to_np(out)
            q = self.plot_image_grid([np.clip(out_tmp, 0, 1), img_np], factor=13)

            shape = shapesOrigin[currentImage]
            out_resize = tvt.Resize(shape, antialias=True)(out) # Apply antialiasing for bilinear or bicubic modes
            out_np = self.torch_to_np(out_resize)
            
            out_np = out_np.mean(axis=0)
            weight = weightsOrigin[currentImage]
            out_np *= weight/out_np.sum()
            
            infoPath = fname.split("."+imageSaveType)[-2] + ".info"
            txtPath = fname.split("."+imageSaveType)[-2] + f"_{self.i}iteration_deno.txt"
            np.savetxt(txtPath, out_np, delimiter="\t") # 3 channels --> 1 channel
            self.Regenerate_toITX(infoPath, txtPath, self.i)


if __name__ == '__main__':

    app = ttk.Window("Peco", "vapor") # solar, vapor
    call = denoiseEngine(app)
    app.mainloop()

