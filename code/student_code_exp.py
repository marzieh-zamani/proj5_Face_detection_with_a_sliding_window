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
    # positive_files = positive_files[:10]

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    
    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(len(positive_files), n_cell*n_cell*31)
    feats_flip = np.random.rand(len(positive_files), n_cell*n_cell*31)
    feats_wrap = np.random.rand(3*len(positive_files), n_cell*n_cell*31)

    # angles and scales for warping
    angle_list = [10, 20, 30]
    scale_list = [1.2, 1.3, 1.4]
    Nw = len(angle_list)



    # Load images
    for idx, path in enumerate(positive_files):

        im = load_image_gray(path)

        # HOF features for normal image
        im_feats = vlfeat.hog.hog(im, cell_size)
        feats[idx,:] = im_feats.flatten() 
        # img = (img - np.mean(img))/np.std(img)

        # change image contrast
        # contrast   = 1
        # brightness = 0.5
        # img = cv2.addWeighted(img, contrast, img, 0, brightness)

        # add noise
        # noise = np.empty(im.shape, np.uint8)
        # cv2.randu(noise,(0),(20))   
        # im = im + noise

        # HOF features for mirrored image
        im_flip = cv2.flip(im, 1)
        im_flip_feats = vlfeat.hog.hog(im_flip, cell_size)
        feats_flip[idx, :] = im_flip_feats.flatten()

        # HOF features for warped image
        for idx_a, angle in enumerate(angle_list):
            im_center = (im.shape[1]//2, im.shape[0]//2)
            rot_mat = cv2.getRotationMatrix2D(im_center, angle, scale_list[idx_a])
            im_warp = cv2.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR)
            im_warp_feats = vlfeat.hog.hog(im_warp, cell_size)
            feats_wrap[Nw*idx+idx_a, :] = im_warp_feats.flatten()

    # print results info
    print('******* get_positive_features info *******')
    print('\n\nExtracted feats from normal image feats.shape: ', feats.shape)
    print('\n\nExtracted feats from flipped image feats_flip.shape: ', feats_flip.shape)
    print('\n\nExtracted feats from warped image feats_warp.shape: ', feats_wrap.shape)

    features_pos_multi = np.concatenate((feats, feats_flip, feats_wrap),axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


    return feats, features_pos_multi

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
    feats_multi_scale = np.random.rand(num_samples * 10, n_cell * n_cell * 31)

    k_single = -1
    k_multi = -1
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
        scale_list = [1, 0.8, 0.65] #[random.randint(70,100)/100]

        for scale_factor in scale_list:
            im = cv2.resize(im, dsize=None, fx=scale_factor, fy=scale_factor)

            # step_sizewin_size
            for cur_y_min in range(0, im.shape[0] - win_size, win_size):
                for cur_x_min in range(0, im.shape[1] - win_size, win_size):

                    #extract feature for current bounding box
                    im_c = im[cur_y_min : cur_y_min + win_size, cur_x_min : cur_x_min + win_size]
                    im_feats = vlfeat.hog.hog(im_c, cell_size)

                    # save feats to feats_multi_scale
                    k_multi = k_multi + 1
                    feats_multi_scale[k_multi,:] = im_feats.flatten()

                    # save feats to feats
                    if scale_factor == 1:
                        k_single = k_single + 1
                        feats[k_single,:] = im_feats.flatten()


    # check feats size
    feats = feats[:k_single]

    if k_single > num_samples:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        feats = feats[indices,:]

    # check feats_multi_scale size
    feats_multi_scale = feats_multi_scale[:k_multi]
    if k_multi > num_samples:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        feats_multi_scale = feats_multi_scale[indices,:]

    # print results info
    print('******* get_random_negative_features info *******')
    print('\n\nExtracted feats from normal image feats.shape: ', feats.shape)
    print('\n\nExtracted feats from multi-scale image feats_multi_scale.shape: ', feats_multi_scale.shape)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats, feats_multi_scale

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
    # LinearSVC(penalty='l2', loss='squared_hinge', *, dual=True, tol=0.0001, 
    # C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, 
    # class_weight=None, verbose=0, random_state=None, max_iter=1000)

    clf = LinearSVC(C = C, max_iter=10000)
    # clf = LinearSVC(random_state=0, tol=1, C=C, max_iter=1000)

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
    # negative_files = negative_files[:10]

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size/cell_size).astype('int')
    # feats = np.random.rand(len(negative_files), n_cell*n_cell*31)
    feats = np.random.rand(10*len(negative_files), n_cell*n_cell*31)
    feats_multi_scale = np.random.rand(10*len(negative_files), n_cell*n_cell*31)

    k_single = -1
    k_multi = -1
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
        scale_list = [1, 0.8, 0.65] #[random.randint(70,100)/100]

        for scale_factor in scale_list:
            im = cv2.resize(im, dsize=None, fx=scale_factor, fy=scale_factor)

            # step_sizewin_size
            for cur_y_min in range(0, im.shape[0] - win_size, win_size):
                for cur_x_min in range(0, im.shape[1] - win_size, win_size):

                    #extract feature for current bounding box
                    im_c = im[cur_y_min : cur_y_min + win_size, cur_x_min : cur_x_min + win_size]
                    im_feats = vlfeat.hog.hog(im_c, cell_size)

                    im_pred = svm.predict(im_feats.reshape(1, -1))

                    if im_pred == 1:
                        # save feats to feats_multi_scale
                        k_multi = k_multi + 1
                        feats_multi_scale[k_multi,:] = im_feats.flatten()

                        # save feats to feats
                        if scale_factor == 1:
                            k_single = k_single + 1
                            feats[k_single,:] = im_feats.flatten()

    # check feats size
    feats = feats[:k_single]

    # check feats_multi_scale size
    feats_multi_scale = feats_multi_scale[:k_multi]

    # print results info
    # print('******* get_random_negative_features info *******')
    # print('\n\nExtracted feats from normal image feats.shape: ', feats.shape)
    # print('\n\nExtracted feats from multi-scale image feats_multi_scale.shape: ', feats_multi_scale.shape)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats, feats_multi_scale

def run_detector(test_scn_path, svm, feature_params, scale_id, verbose=False):
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
    # im_filenames = im_filenames[:10]
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
    if scale_id == 0:
        # print('run_detector for single scale')
        multi_scale_factor = [1]
    else:
        # print('run_detector for multi scale')
        multi_scale_factor = np.array([1, 0.8, 0.65, 0.5, 0.3, 0.25])

    # k = -1
    for idx, im_filename in enumerate(im_filenames):
        # print('Detecting faces in {:s}'.format(im_filename))
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
        # multi_scale_factor = np.array([1, 0.8, 0.65, 0.5, 0.3, 0.25])
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

        # print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))

        if sum(is_valid_bbox) == 0:
            # print('*** continue ***')
            continue
        
        cur_bboxes = cur_bboxes[is_valid_bbox]
        cur_confidences = cur_confidences[is_valid_bbox]

        bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    return bboxes, confidences, image_ids