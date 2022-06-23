from pyexpat.errors import XML_ERROR_NOT_STANDALONE
import numpy as np

from scipy.ndimage import interpolation


class PreProcess(object):
    """Various methods to preprocess with preprocess() applying all of them.
    """
    def __init__(self, x_train,y_train,x_test,y_test ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def _moments(self,image):
        c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
        totalImage = np.sum(image) #sum of pixels
        m0 = np.sum(c0*image)/totalImage #mu_x
        m1 = np.sum(c1*image)/totalImage #mu_y
        m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
        m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
        m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
        mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
        covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
        return mu_vector, covariance_matrix
    #Deskew the training samples 
    def deskew(self, image):
        c,v = self._moments(image)
        alpha = v[0,1]/v[0,0]
        affine = np.array([[1,0],[alpha,1]])
        ocenter = np.array(image.shape)/2.0
        offset = c-np.dot(affine,ocenter)
        img = interpolation.affine_transform(image,affine,offset=offset)
        return (img - img.min()) / (img.max() - img.min())
        
    def preprocess(self):
        #training set 
        x_train_deskew = [] 
        for i in range(self.x_train.shape[0]): 
            x_train_deskew.append(self.deskew(self.x_train[i].reshape(28,28)))
        x_train_deskew = np.array(x_train_deskew)
        x_train_deskew = x_train_deskew[..., np.newaxis]
        print("shape of x_train_deskew is " + str(np.shape(x_train_deskew)))
        print("type of x_train_deskew is " + str(type(x_train_deskew)))

        #test set 
        x_test_deskew = [] 
        for j in range(self.x_test.shape[0]): 
            x_test_deskew.append(self.deskew(self.x_test[j].reshape(28,28)))
        x_test_deskew = np.array(x_test_deskew)
        x_test_deskew = x_test_deskew[..., np.newaxis]
        print("shape of x_test_deskew is " + str(np.shape(x_test_deskew)))
        print("type of x_test_deskew is " + str(type(x_test_deskew)))
        return x_train_deskew, x_test_deskew

