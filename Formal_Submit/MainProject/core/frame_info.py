#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen

import numpy as np

def image_normalize(im, axis = (0,1), c = 1e-8):
    '''
    Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)
   
 
# Each area (ndvi, pan, annotation, weight) is represented as an Frame
class FrameInfo:
    """ Defines a frame, includes its constituent images, annotation and weights (for weighted loss).
    """

    def __init__(self, img, annotations, weight, dtype=np.float32):
        """FrameInfo constructor.

        Args:
            img: ndarray
                3D array containing various input channels.
            annotations: ndarray
                3D array containing human labels, height and width must be same as img.
            weight: ndarray
                3D array containing weights for certain losses.
            dtype: np.float32, optional
                datatype of the array.
        """
        self.img = img
        self.annotations = annotations
        self.weight = weight
        self.dtype = dtype

    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def getPatch(self, i, j, patch_size, img_size, normalize=1.0): #改
        """Function to get patch from the given location of the given size.

        Args:
            i: int
                Starting location on first dimension (x axis).
            y: int
                Starting location on second dimension (y axis).
            patch_size: tuple(int, int)
                Size of the patch.
            img_size: tuple(int, int)
                Total size of the images from which the patch is generated.
        """
        patch = np.zeros(patch_size, dtype=self.dtype)
        img_shape = self.img.shape
        # print('img_shape=' + str(img_shape))
        # print('img_shape[0]=' + str(img_shape[0]))
        # print('img_shape[1]=' + str(img_shape[1]))
        # print('img_shape[2]=' + str(img_shape[2]))
        # print("img_size[0]"+str(img_size[0]))
        # print("img_size[1]" + str(img_size[1]))
        #print("img_size[2]" + str(img_size[2])) # out of range
        #
        # print("self.img[0]" + str(np.shape(self.img[0])))
        # print("self.annotations" + str(np.shape(self.annotations)))
        # print("self.weight" + str(np.shape(self.weight)))
        #
        # print("patch_size[0]"+str(patch_size[0]))
        # print("patch_size[1]"+str(patch_size[1]))
        # print("patch_size[2]"+str(patch_size[2]))

        # print("min(img_shape[1], patch_size[0]" + str(min(img_shape[1], patch_size[0])))
        # print("min(img_shape[2], patch_size[1]" + str(min(img_shape[2], patch_size[1])))

        #print("img_size[2]" + str(img_size[2])) # out of range
        # print("img_size.shape"+str(np.shape(img_size)))
        #print("img_size[2]" + str(img_size[2]))
        # print("patch_size" + str(patch_size))
        # print("self.img[0]" + str(np.shape(self.img[0])))
        #im = self.img[i:i + img_size[0], j:j + img_size[1]]  # 这玩意儿还是个三波段的东西
        #im = self.img[i:i + img_size[0], j:j + img_size[1]] #改
        #print("O oringinal im" + str(np.shape(im)))
        im1 = self.img[0][i:i + img_size[0], j:j + img_size[1]]
        im2 = self.img[1][i:i + img_size[0], j:j + img_size[1]]
        im3 = self.img[2][i:i + img_size[0], j:j + img_size[1]]
        r = np.random.random(1)
        if normalize >= r[0]:
            im1 = image_normalize(im1, axis=(0, 1)) #改
            im2 = image_normalize(im2, axis=(0, 1))
            im3 = image_normalize(im3, axis=(0, 1))
        #an = self.annotations[i:i + img_size[0], j:j + img_size[1], k:k + img_size[2]] #改
        an = self.annotations[i:i + img_size[0], j:j + img_size[1]]
        #yy1 = yy.transpose(1, 2, 0)
        an = np.expand_dims(an, axis=-1)
        #we = self.weight[i:i + img_size[0], j:j + img_size[1], k:k + img_size[2]] #改
        #print(np.shape(self.weight))
        we = self.weight[i:i + img_size[0], j:j + img_size[1]]
        we = np.expand_dims(we, axis=-1)
        # print("original im"+str(np.shape(self.img)))
        # print("original an"+str(np.shape(self.annotations)))
        # print("original we"+str(np.shape(self.weight)))
        #print("make patch im" + str(np.shape(im)))
        #print("make patch im1"+str(np.shape(im1)))
        im1 = im1[np.newaxis, :, :]
        #print("add D im1" + str(np.shape(im1)))
        im2 = im2[np.newaxis, :, :]
        #print("add D im2" + str(np.shape(im2)))
        im3 = im3[np.newaxis, :, :]
        # print("add D im3" + str(np.shape(im3)))
        #
        # print("make patch im2" + str(np.shape(im2)))
        # print("make patch im3" + str(np.shape(im3)))
        #im = np.concatenate(im1, im2, im3)
        # print("final im " + str(np.shape(im)))
        an = an.transpose(2, 0, 1)
        we = we.transpose(2, 0, 1)
        #print("make patch an"+str(np.shape(an)))
        #print("make patch we"+str(np.shape(we)))
        #print(im.)
        #comb_img = np.concatenate((im, an, we), axis=-1)
        comb_img = np.concatenate((im1, im2, im3, an, we), axis=0)
        comb_img = comb_img.transpose(1,2,0)
        #print("comb_img" + str(np.shape(comb_img)))
        patch[:img_size[0], :img_size[1],] = comb_img  # 单波段
        #patch[:img_size[0], :img_size[1], :img_size[2]] =  comb_img #单波段
        #patch[:img_size,] = comb_img
        return (patch)

    # Returns all patches in a image, sequentially generated
    def sequential_patches(self, patch_size, step_size, normalize):
        """All sequential patches in this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            step_size: tuple(int, int)
                Total size of the images from which the patch is generated.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img.shape
        #print('img_shape='+img_shape)
        #x = range(0, img_shape[0] - patch_size[0], step_size[0])
        #y = range(0, img_shape[1] - patch_size[1], step_size[1])
        x = range(0, img_shape[1] - patch_size[0], step_size[0])
        y = range(0, img_shape[2] - patch_size[1], step_size[1])
        #z = range(0, img_shape[2] - patch_size[2], step_size[2])
        # = range(0, img_shape[3] - patch_size[3], step_size[3])#改
        if (img_shape[1] <= patch_size[0]):
            x = [0]
        if (img_shape[2] <= patch_size[1]):
            y = [0]
        # if (img_shape[2] <= patch_size[2]): #改
        #     z = [0]
        ic = (min(img_shape[1], patch_size[0]), min(img_shape[2], patch_size[1]))  # 改
        #ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))  # 改
        #ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]), min(img_shape[2], patch_size[2])) #改
        xy = [(i, j) for i in x for j in y]  # 改
        #xyz = [(i, j, k) for i in x for j in y for k in z] #改
        img_patches = []
        for i, j in xy: # 改
            img_patch = self.getPatch(i, j, patch_size, ic, normalize) # 改
            img_patches.append(img_patch)
        #print(len(img_patches))
        return (img_patches)

    # Returns a single patch, startring at a random image
    def random_patch(self, patch_size, normalize):
        """A random from this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img.shape
        if (img_shape[1] <= patch_size[0]):
            x = 0
        else:
            x = np.random.randint(0, img_shape[1] - patch_size[0])
        if (img_shape[2] <= patch_size[1]):
            y = 0
        else:
            y = np.random.randint(0, img_shape[2] - patch_size[1])
        #改
        # if (img_shape[2] <= patch_size[2]):
        #     z = 0
        # else:
        #     z = np.random.randint(0, img_shape[2] - patch_size[2])  # 改
        # 改
        ic = (min(img_shape[1], patch_size[0]), min(img_shape[2], patch_size[1]))
        #ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]), min(img_shape[2], patch_size[2]))# 改
        img_patch = self.getPatch(x, y, patch_size, ic, normalize) # 改
        return (img_patch)
