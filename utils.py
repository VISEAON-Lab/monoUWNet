from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import numpy as np
import cv2
from my_utils import toNumpy
import random

class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.
    https://github.com/glasgio/homomorphic-filter/blob/eacc5d236ee2f15a40db120fd16d8221d61859bf/homofilt.py#L5
    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H
        
        .
    """

    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/(filter_params[0]+1e-3)**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H = None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) != 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return (I)
# End of class HomomorphicFilter


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


water_types_Nrer_rgb = {}
water_types_Nrer_rgb["I"] = np.exp(-np.array([0.233, 0.049, 0.021]))
water_types_Nrer_rgb["IA"] = np.exp(-np.array([0.234,  0.0503, 0.0253]))
water_types_Nrer_rgb["3C"] = np.exp(-np.array([0.380,  0.187, 0.240]))

def estimateA(img, depth):
    # finding BL
    p = np.percentile(depth, 99.9)
    depth_10p = depth.copy()
    depth_10p[depth_10p<p]=0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_BL = gray.copy()
    img_BL[depth_10p<p]=0
    rmax, cmax = np.unravel_index(img_BL.argmax(), img_BL.shape)
    BL = img[rmax, cmax, :]
    return BL




def computeJ(image, depth):
    img = toNumpy(image)
    depth = toNumpy(depth)
    A = estimateA(img, depth)
    TM = np.zeros_like(img)
    for t in range(3):
        # TM[:,:,t] =  np.exp(-beta_rgb[t]*depth)
        TM[:,:,t] =  water_types_Nrer_rgb["3C"][t]**depth
    S = A*(1-TM)
    J = (img - A) / TM + A
    return J # TODO: convert back to pytorch

def homorphicFiltering(img, G=None, x=None):

    img = np.float32(img)
    img = img/255

    rows,cols,dim=img.shape

    rh, rl, cutoff = 0.6,0.5,32

    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y,cr,cb = cv2.split(imgYCrCb)

    # y_log = np.log(y+0.01)

    # y_fft = np.fft.fft2(y_log)

    # y_fft_shift = np.fft.fftshift(y_fft)


    # DX = cols/cutoff
    # if G is None:
    #     G = np.ones((rows,cols))
    #     for i in range(rows):
    #         for j in range(cols):
    #             G[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl
    
    # result_filter = G * y_fft_shift
    # result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))
    # result = np.exp(result_interm)
    # result = result.astype(np.float32)
    # rgb = np.dstack((result,cr,cb)) 

    # homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
    homo_filter = HomomorphicFilter()
    filter_params=[5,2]
    # random HF!!
    # if x is None: // change every image from now on
    x = random.randrange(1,20)*5

    filter_params[0] = x
    img_filtered = homo_filter.filter(I=y, filter_params=filter_params).astype(np.float32)
    y_max = np.max(y)
    y_min = np.min(y)
    img_filtered_max = np.max(img_filtered)
    img_filtered_min = np.min(img_filtered)
    z = (img_filtered - img_filtered_min) / (img_filtered_max - img_filtered_min)
    z = z*(y_max - y_min) + y_min
    rgb = np.dstack((img_filtered,cr,cb)) 
    rgb = cv2.cvtColor(rgb, cv2.COLOR_YCrCb2RGB)
    rgb[rgb<0]=0
    rgb*=255
    rgb  = rgb.astype(np.uint8)
    return rgb
