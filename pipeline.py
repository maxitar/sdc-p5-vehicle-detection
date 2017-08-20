import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
from moviepy.editor import VideoFileClip
from sklearn.utils import shuffle
import time
import argparse
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
import helper_functions as hf

class HogParams():
    def __init__(self, orientations, pixels_per_cell, cells_per_block):
        self.orient = orientations
        self.pix_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

class Tracker():
    def __init__(self, svm, X_scaler, hog_params, y_ranges, win_sizes, debug):
        self.svm = svm
        self.scaler = X_scaler
        self.hog_params = hog_params
        self.y_ranges = y_ranges
        self.win_sizes = win_sizes
        self.debug = debug
        self.heatmaps = []

    def get_boxes(self, img):
        img_width = img.shape[1]
        all_boxes = []
        predictions = []
        use_two_colors = False
        img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if use_two_colors:
            img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

        for y_range, win_size in zip(self.y_ranges, self.win_sizes):
            xscale = win_size[0]/64
            yscale = win_size[1]/64

            img_ycc_region = img_ycc[y_range[0]:y_range[1],:,:]
            imshape = img_ycc_region.shape
            if win_size[0] != 64 or win_size[1] != 64:
                img_ycc_region = cv2.resize(img_ycc_region, (np.int(imshape[1]/xscale), np.int(imshape[0]/yscale)))
            ch1_ycc = img_ycc_region[:,:,0]
            ch2_ycc = img_ycc_region[:,:,1]
            ch3_ycc = img_ycc_region[:,:,2]
            hog1 = hf.get_hog_features(ch1_ycc, self.hog_params.orient, self.hog_params.pix_per_cell, self.hog_params.cells_per_block, feature_vec=False)
            hog2 = hf.get_hog_features(ch2_ycc, self.hog_params.orient, self.hog_params.pix_per_cell, self.hog_params.cells_per_block, feature_vec=False)
            hog3 = hf.get_hog_features(ch3_ycc, self.hog_params.orient, self.hog_params.pix_per_cell, self.hog_params.cells_per_block, feature_vec=False)

            if use_two_colors:
                img_luv_region = img_luv[y_range[0]:y_range[1],:,:]
                if win_size[0] != 64 or win_size[1] != 64:
                    img_luv_region = cv2.resize(img_luv_region, (np.int(imshape[1]/xscale), np.int(imshape[0]/yscale)))
                ch1_luv = img_luv_region[:,:,0]
                ch2_luv = img_luv_region[:,:,1]
                ch3_luv = img_luv_region[:,:,2]
                hog4 = hf.get_hog_features(ch1_luv, self.hog_params.orient, self.hog_params.pix_per_cell, self.hog_params.cells_per_block, feature_vec=False)
                hog5 = hf.get_hog_features(ch2_luv, self.hog_params.orient, self.hog_params.pix_per_cell, self.hog_params.cells_per_block, feature_vec=False)
                hog6 = hf.get_hog_features(ch3_luv, self.hog_params.orient, self.hog_params.pix_per_cell, self.hog_params.cells_per_block, feature_vec=False)


            # Define blocks and steps as above
            nxblocks = (ch1_ycc.shape[1] // self.hog_params.pix_per_cell) - self.hog_params.cells_per_block + 1
            nyblocks = (ch1_ycc.shape[0] // self.hog_params.pix_per_cell) - self.hog_params.cells_per_block + 1
            nfeat_per_block = self.hog_params.orient*self.hog_params.cells_per_block**2

            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // self.hog_params.pix_per_cell) - self.hog_params.cells_per_block + 1
            cells_per_step = 1  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_features = np.array([], dtype=hog1.dtype)
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    if use_two_colors:
                        hog_feat1 = hog4[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat2 = hog5[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat3 = hog6[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_features = np.hstack((hog_features, hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos*self.hog_params.pix_per_cell
                    ytop =  ypos*self.hog_params.pix_per_cell

                    xbox_left = np.int(xleft*xscale)
                    ytop_draw = np.int(ytop*yscale)+y_range[0]
                    all_boxes.append([[xbox_left,ytop_draw],[xbox_left+win_size[0], ytop_draw+win_size[1]]])
                    self.scaler.transform(hog_features)
                    hog_features = hog_features.reshape(1,-1)
                    predictions.append(self.svm.predict(hog_features)[0]==1)

        return all_boxes, predictions

    def process_frame(self, img):
        all_boxes, predictions = self.get_boxes(img)
        draw_img = img.copy()
        car_boxes_labeled = []
        car_preds = np.array(predictions)
        all_boxes = np.array(all_boxes, dtype=int)
        car_boxes = all_boxes[car_preds]

        heatmap = np.zeros_like(draw_img[:,:,0], dtype=float)
        hf.add_heat(heatmap, car_boxes)
        self.heatmaps.append(heatmap)
        keep_frames = 15
        active_frames = 11
        self.heatmaps = self.heatmaps[-keep_frames:]
        # Heat places that were in more than 2/3rds of the last 15 frames
        heatmap_med15 = np.sum(np.bool_(self.heatmaps), axis=0)//active_frames
        # but get strong detections faster
        heatmap_strong = np.mean(self.heatmaps[-2:], axis=0)

        if self.debug == False:
            # Only leave places that have more than 5 boxes on avg between the last two frames
            hf.apply_threshold(heatmap_strong, 5)
            heatmap = heatmap_strong+heatmap_med15
            labels = label(heatmap)
            car_boxes_labeled = hf.get_labeled_bboxes(labels)
            for box in car_boxes_labeled:
                cv2.rectangle(draw_img, pt1=(box[0][0],box[0][1]), pt2=(box[1][0],box[1][1]), color=(0,255,0), thickness=3)
        else:
            # If debugging split the frame into four regions
            # Top left region is the image with all the sliding windows in green, and the detections in blue
            for box in all_boxes:
                cv2.rectangle(draw_img, pt1=(box[0][0],box[0][1]), pt2=(box[1][0],box[1][1]), color=(0,255,0), thickness=2)
            for box in car_boxes:
                cv2.rectangle(draw_img, pt1=(box[0][0],box[0][1]), pt2=(box[1][0],box[1][1]), color=(0,0,255), thickness=2)
            outimg = np.zeros_like(img)
            half_height = img.shape[0]//2
            half_width = img.shape[1]//2
            outimg[:half_height,:half_width] = cv2.resize(draw_img, (half_width, half_height))
            # Bottom left region is the heatmap, before the threshold
            heatmap = heatmap_strong+heatmap_med15
            max_heat = max(1, np.max(heatmap))
            heat_vis = np.uint8(255*heatmap/max_heat)
            outimg[half_height:,:half_width,0] = cv2.resize(heat_vis, (half_width, half_height))
            # Bottom right region is the heatmap, after the threshold
            hf.apply_threshold(heatmap_strong, 5)
            heatmap = heatmap_strong+heatmap_med15
            labels = label(heatmap)
            heat_vis = np.uint8(255*heatmap/max_heat)
            heat_vis[heat_vis>0] = 255
            outimg[half_height:,half_width:,0] = cv2.resize(heat_vis, (half_width, half_height))
            # Top right region is the final image
            car_boxes_labeled = hf.get_labeled_bboxes(labels)
            draw_img = img.copy()
            for box in car_boxes_labeled:
                cv2.rectangle(draw_img, pt1=(box[0][0],box[0][1]), pt2=(box[1][0],box[1][1]), color=(0,255,0), thickness=3)
            outimg[:half_height,half_width:] = cv2.resize(draw_img, (half_width, half_height))

            cv2.line(outimg, (0, half_height), (outimg.shape[1], half_height), (255,255,255), 2)
            cv2.line(outimg, (half_width, 0), (half_width, outimg.shape[0]), (255,255,255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(outimg,'Windows and detections',(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(outimg,'Final result',(10+half_width,50), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(outimg,'Heatmap before threshold',(10,50+half_height), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(outimg,'Heatmap after threshold',(10+half_width,50+half_height), font, 1,(255,255,255),2,cv2.LINE_AA)
            draw_img = outimg
        return draw_img

def get_data(orientations, pixels_per_cell, cells_per_block):
    print('Reading image names...')
    v_train_names, v_test_names, nv_train_names, nv_test_names = hf.read_images(train_split = 0.8)
    img_train_names = v_train_names+nv_train_names
    img_test_names = v_test_names+nv_test_names
    print('Read {} vehicle and {} non-vehicle training images'.format(len(v_train_names), len(nv_train_names)))
    print('Read {} vehicle and {} non-vehicle test images'.format(len(v_test_names), len(nv_test_names)))
    print('Reading images and finding features...')
    X_train = []
    for img_name in img_train_names:
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_features = hf.get_features(img, orientations, pixels_per_cell, cells_per_block)
        X_train.append(img_features)
    X_train = np.array(X_train)

    X_test = []
    for img_name in img_test_names:
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_features = hf.get_features(img, orientations, pixels_per_cell, cells_per_block)
        X_test.append(img_features)
    X_test = np.array(X_test)

    y_train = np.zeros((X_train.shape[0]))
    y_train[:len(v_train_names)] = 1
    y_test = np.zeros((X_test.shape[0]))
    y_test[:len(v_test_names)] = 1
    return X_train, y_train, X_test, y_test

def pipeline_train(orientations, pixels_per_cell, cells_per_block):
    start_time = time.perf_counter()
    X_train, y_train, X_test, y_test = get_data(orientations, pixels_per_cell, cells_per_block)
    print('Time for obtaining image features is {:.2f}s'.format(time.perf_counter()-start_time))
    print('Shape of X_train: {}'.format(X_train.shape))
    print('Scaling data...')
    start_time = time.perf_counter()
    X_scaler = StandardScaler()
    X_scaler.fit_transform(X_train)
    X_train, y_train = shuffle(X_train, y_train)
    X_scaler.transform(X_test)
    print('Time for scaling data is {:.2f}s'.format(time.perf_counter()-start_time))
    print('Training svm...')
    start_time = time.perf_counter()
    svm = LinearSVC(C=0.5, tol=1e-5, verbose=1)
    svm.fit(X_train, y_train)
    print('Time for svm training is {:.2f}s'.format(time.perf_counter()-start_time))
    print('SVM train accuracy {:.2f}%'.format(svm.score(X_train, y_train)*100))
    print('SVM test accuracy {:.2f}%'.format(svm.score(X_test, y_test)*100))
    return svm, X_scaler

def get_output(proc, args):
    ext = args.name[-3:]
    if ext == 'mp4':
        if args.debug:
            video_output = 'video_output_debug.mp4'
        else:
            video_output = 'video_output.mp4'
        clip2 = VideoFileClip(args.name).subclip(args.movie_start, args.movie_end)#.subclip(20,30)#.subclip(10,15)#.subclip(20,25)
        output_clip = clip2.fl_image(proc.process_frame)
        output_clip.write_videofile(video_output, audio=False)
    else:
        img = mpimg.imread(args.name)
        if img.shape[2] == 4:
            img = img[:,:,:3]
        if args.debug:
            image_output = 'image_output_debug.jpg'
        else:
            image_output = 'image_output.jpg'
        out_img = proc.process_frame(img)
        mpimg.imsave(image_output, out_img)
        plt.imshow(out_img)
        plt.show()

if __name__ == '__main__':
    orientations = 10
    pixels_per_cell = 16
    cells_per_block = 3
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="the name of file for processing")
    parser.add_argument("-s","--movie_start", type=int, default=0, help="if the file is a movie, specify starting second")
    parser.add_argument("-e","--movie_end", type=int, default=None, help="if the file is a movie, specify last second")
    parser.add_argument("-d","--debug", default=False, action="store_true", help="save output in debug mode")
    args = parser.parse_args()
    if not os.path.exists(args.name):
        raise ValueError('File {} does not exist'.format(args.name))
    if not os.path.exists('svm.p'):
        svm, X_scaler = pipeline_train(orientations, pixels_per_cell, cells_per_block)
        pardict = {}
        pardict['svm'] = svm
        pardict['scaler'] = X_scaler
        pardict['orientations'] = orientations
        pardict['pixels_per_cell'] = pixels_per_cell
        pardict['cells_per_block'] = cells_per_block
        with open('svm.p', 'wb') as pickle_file:
            pickle.dump(pardict, pickle_file)
    else:
        with open('svm.p', 'rb') as pickle_file:
            pardict = pickle.load(pickle_file)
        svm = pardict['svm']
        X_scaler = pardict['scaler']
        orientations = pardict['orientations']
        pixels_per_cell = pardict['pixels_per_cell']
        cells_per_block = pardict['cells_per_block']

    hog_params = HogParams(orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
    y_ranges=([400,560],[400,656],[432,656])
    win_sizes = ([72,72],[96,96],[128,128])
    proc = Tracker(svm, X_scaler, hog_params, y_ranges, win_sizes, args.debug)

    get_output(proc, args)
