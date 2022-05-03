import numpy as np
import cyvlfeat as vlfeat
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC
import random


def get_positive_features(train_path_pos, feature_params):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    
    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(2*len(positive_files), n_cell*n_cell*31)

    # angle_list = [0, 10, 20, 30]

    # Load images
    for idx, path in enumerate(positive_files):

        im = load_image_gray(path)
        # img = (img - np.mean(img))/np.std(img)

        # change image contrast
        # contrast   = 1
        # brightness = 0.5
        # img = cv2.addWeighted(img, contrast, img, 0, brightness)

        # add noise
        # noise = np.empty(im.shape, np.uint8)
        # cv2.randu(noise,(0),(20))   
        # im = im + noise

        im_flip = cv2.flip(im, 1)

        im_feats = vlfeat.hog.hog(im, cell_size)
        im_flip_feats = vlfeat.hog.hog(im_flip, cell_size)

        feats[2*idx,:] = im_feats.flatten() 
        feats[2*idx+1, :] = im_flip_feats.flatten()

        # for ia, ang in enumerate(angle_list):
        #     image_center = tuple(np.array(img.shape[1::-1]) / 2)
        #     rot_mat = cv2.getRotationMatrix2D(image_center, ang, 1.0)
        #     img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)



        #     img_mirrored = cv2.flip(img, 1)

        #     img_hog_feats = vlfeat.hog.hog(img, cell_size)
        #     img_mirrored_hog_feats = vlfeat.hog.hog(img_mirrored, cell_size)

        #     feats[8*i+ia,:] = img_hog_feats.flatten() 
        #     feats[8*i+ia+4, :] = img_mirrored_hog_feats.flatten()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    This function should return negative training examples (non-faces) from any
    images in 'non_face_scn_path'. Images should be loaded in grayscale because
    the positive training data is only available in grayscale (use
    load_image_gray()).

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))
    # negative_files = negative_files[:10]

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size/cell_size).astype('int')
#     feats = np.random.rand(len(negative_files), n_cell*n_cell*31)
    feats = np.random.rand(num_samples * 10, n_cell * n_cell * 31)
    print(feats.shape)

    k = -1
    for path in negative_files:
        im = load_image_gray(path)
        # im = (im - np.mean(im))/np.std(im)

        # # add noise
        # noise = np.empty(im.shape, np.uint8)
        # cv2.randu(noise,(0),(20))   
        # im = im + noise

        # Define list of scale values 
        # scale_factor_list = list([0.1*round(3600/max(im.shape))]) # [1.0, 0.8, 0.65] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # 
        # scale_factor_list = [1.0, 0.8, 0.65]
        scale_factor_list = [1] #[random.randint(70,100)/100]

        for scale_factor in scale_factor_list:
            # img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor)

            # step_sizewin_size
            for cur_y_min in range(0, im.shape[0] - win_size, win_size):
                for cur_x_min in range(0, im.shape[1] - win_size, win_size):

                    #extract feature for current bounding box
                    im_c = im[cur_y_min : cur_y_min + win_size, cur_x_min : cur_x_min + win_size]
                    im_feats = vlfeat.hog.hog(im_c, cell_size)
                    k = k+1
                    feats[k,:] = im_feats.flatten() #np.reshape(hog_feats,(1, ((win_size//cell_size)**2)*31))

    # 
    feats = feats[:k]
    print('negative_files', len(negative_files), 'k', k)
    print(feats.shape)
    # num_samples = 100

    # indices = np.random.randint(0,k, num_samples)
    # arr = np.arange(10)
    if k > num_samples:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        feats = feats[indices,:]
        print(feats.shape)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def train_classifier(features_pos, features_neg, C):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """
    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    # svm = PseudoSVM(10,features_pos.shape[1])

    # Construct x_train & y_train
    x_train = np.vstack((features_pos,features_neg))
    y_train = np.ones((x_train.shape[0]))
    y_train[features_pos.shape[0]:] = -1

    # Initialize LinearSVC
    # clf = LinearSVC(C = C)
    clf = LinearSVC(random_state=0, tol=1, C=C)

    # fit to train data
    svm = clf.fit(x_train,y_train)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    -   svm.predict(feat): predict features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size/cell_size).astype('int')
    # feats = np.random.rand(len(negative_files), n_cell*n_cell*31)
    feats = np.random.rand(10*len(negative_files), n_cell*n_cell*31)
    print(feats.shape)

    k = -1
    for path in negative_files:
        im = load_image_gray(path)
        # im = (im - np.mean(im))/np.std(im)

        # # add noise
        # noise = np.empty(im.shape, np.uint8)
        # cv2.randu(noise,(0),(20))   
        # im = im + noise

        # Define list of scale values 
        # scale_factor_list = list([0.1*round(3600/max(im.shape))]) # [1.0, 0.8, 0.65] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # 
        # scale_factor_list = [1.0, 0.8, 0.65]
        scale_factor_list = [1] #[random.randint(70,100)/100]

        for scale_factor in scale_factor_list:
            # img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor)

            # step_size win_size
            for cur_y_min in range(0, im.shape[0] - win_size, win_size):
                for cur_x_min in range(0, im.shape[1] - win_size, win_size):

                    #extract feature for current bounding box
                    im_c = im[cur_y_min : cur_y_min + win_size, cur_x_min : cur_x_min + win_size]
                    im_feats = vlfeat.hog.hog(im_c, cell_size)

                    im_pred = svm.predict(im_feats.reshape(1, -1))

                    if im_pred == 1:
                        k = k+1
                        feats[k,:] = im_feats.flatten() #np.reshape(hog_feats,(1, ((win_size//cell_size)**2)*31))

    # 
    feats = feats[:k]
    print(feats.shape)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
    """
    This function returns detections on all of the images in a given path. You
    will want to use non-maximum suppression on your detections or your
    performance will be poor (the evaluation counts a duplicate detection as
    wrong). The non-maximum suppression is done on a per-image basis. The
    starter code includes a call to a provided non-max suppression function.

    The placeholder version of this code will return random bounding boxes in
    each test image. It will even do non-maximum suppression on the random
    bounding boxes to give you an example of how to call the function.

    Your actual code should convert each test image to HoG feature space with
    a _single_ call to vlfeat.hog.hog() for each scale. Then step over the HoG
    cells, taking groups of cells that are the same size as your learned
    template, and classifying them. If the classification is above some
    confidence, keep the detection and then pass all the detections for an
    image to non-maximum suppression. For your initial debugging, you can
    operate only at a single scale and you can skip calling non-maximum
    suppression. Err on the side of having a low confidence threshold (even
    less than zero) to achieve high enough recall.

    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. This function should work for the
            MIT+CMU test set but also for any other images (e.g. class photos).
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slowerbecause the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time.
    -   verbose: prints out debug information if True

    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # number of top detections to feed to NMS
    topk = 50

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)

    # k = -1
    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape
        # create scale space HOG pyramid and return scores for prediction

        #######################################################################
        #                        TODO: YOUR CODE HERE                         #
        #######################################################################

        # cur_x_min = (np.random.rand(15,1) * im_shape[1]).astype('int')
        # cur_y_min = (np.random.rand(15,1) * im_shape[0]).astype('int')
        # cur_bboxes = np.hstack([cur_x_min, cur_y_min, \
        #     (cur_x_min + np.random.rand(15,1)*50).astype('int'), \
        #     (cur_y_min + np.random.rand(15,1)*50).astype('int')])
        # cur_confidences = np.random.rand(15)*4 - 2

        #free parms
        decision_thres = 0
        step_size = 1 #int(cell_size/2)

        cur_bboxes = np.empty((0, 4))
        cur_confidences = np.empty(0)

        # Define list of scale values 
        multi_scale_factor = np.array([1, 0.8, 0.65, 0.5, 0.3, 0.25])
        # multi_scale_factor = np.array([0.9, 0.5, 0.25])
        # min_s = min(360/max(im_shape), 1)
        # min_s = 0.1*round(10*min_s)
        # multi_scale_factor = np.arange(0.1, 2*min_s, 0.2)


        for sf in multi_scale_factor:
            im_s = cv2.resize(im, None, fx = sf, fy = sf, interpolation = cv2.INTER_LINEAR)

            #image to hog feature
            f  = vlfeat.hog.hog(im_s, cell_size)

            #sliding window at multiple scales
            for cur_y_min in range(0, f.shape[0] - template_size, step_size):
                for cur_x_min in range(0, f.shape[1] - template_size, step_size):
                    #extract feature for current bounding box
                    bb_f = f[cur_y_min:cur_y_min+template_size, cur_x_min:cur_x_min+template_size].reshape(1, -1)
                    # bb_f = bb_f.flatten()
                    # bb_f  = bb_f.reshape(1, -1)

                    #classify & threshold classification confidence
                    conf = svm.decision_function(bb_f)
                    if(conf > decision_thres):
                        x1 = int(cur_x_min * cell_size / sf)
                        x2 = int((cur_x_min+template_size) * cell_size / sf)
                        y1 = int(cur_y_min * cell_size / sf)
                        y2 = int((cur_y_min+template_size) * cell_size / sf)
                        bb = np.array([x1, y1, x2, y2])

                        cur_bboxes      = np.vstack((cur_bboxes, bb))
                        cur_confidences = np.hstack((cur_confidences, conf))


        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

        ### non-maximum suppression ###
        # non_max_supr_bbox() can actually get somewhat slow with thousands of
        # initial detections. You could pre-filter the detections by confidence,
        # e.g. a detection with confidence -1.1 will probably never be
        # meaningful. You probably _don't_ want to threshold at 0.0, though. You
        # can get higher recall with a lower threshold. You should not modify
        # anything in non_max_supr_bbox(). If you want to try your own NMS methods,
        # please create another function.

        idsort = np.argsort(-cur_confidences)[:topk]
        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
            im_shape, verbose=verbose)

        print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))

        if sum(is_valid_bbox) == 0:
            print('*** continue ***')
            continue
        
        cur_bboxes = cur_bboxes[is_valid_bbox]
        cur_confidences = cur_confidences[is_valid_bbox]

        bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    return bboxes, confidences, image_ids

# def run_detector(test_scn_path, svm, feature_params, verbose=False):

# #spread out data collection
# #setup for sliding crop window 
# y_max = f.shape[0] - template_size
# x_max = f.shape[1] - template_size

# #sliding window at multiple scales
# for cur_y_min in range(0, y_max, step_size):
#     for cur_x_min in range(0, x_max, step_size):