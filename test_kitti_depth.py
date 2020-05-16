from __future__ import division
import tensorflow as tf
import numpy as np
import os
import cv2
# import scipy.misc
import PIL.Image as pil
from SfMLearner import SfMLearner

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

def main(_):
    # with open('/cmlscratch/arjgpt27/projects/CMSC733/SfMLearnerData/train.txt', 'r') as f:
    folder_name = "./" + FLAGS.output_dir.split('/')[-2] + "/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open('data/kitti/test_files_eigen.txt', 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
        # test_files = [os.path.join(file.split()[0], file.split()[1]) + '.jpg' for file in test_files]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    basename = os.path.basename(FLAGS.ckpt_file)
    output_file = FLAGS.output_dir + '/' + basename
    sfm = SfMLearner()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                fh = open(test_files[idx], 'rb')
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)
                # im = scipy.misc.imread(test_files[idx])
                # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
            pred = sfm.inference(inputs, sess, mode='depth')
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b,:,:,0])
                # cv2.imshow("depth", pred['depth'][b,:,:,0])
                filename = t*100 + b
                # print(pred['depth'][b,:,:,0])
                # exit(-1)
                print(np.min(pred['depth'][b,:,:,0]), np.max(pred['depth'][b,:,:,0]))
                image = pred['depth'][b,:,:,0]*255/np.max(pred['depth'][b,:,:,0])
                image = np.uint8(image)
                # exit(-1)
                cv2.imwrite(folder_name + str(filename) + ".png", image)
                # cv2.imwrite("./original_images/" + str(filename) + ".png", inputs[b,:,:,:])
                # cv2.waitKey(0)
        np.save(output_file, pred_all)

if __name__ == '__main__':
    tf.app.run()