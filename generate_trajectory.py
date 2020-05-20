import glob
import os
import numpy as np
import cv2
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def plot_trajectory(poses_gtruth, directory):

    fig = plt.figure()
    ax = fig.add_subplot(111) 
    prev_pose = poses_gtruth[0]

    for i, pose in enumerate(poses_gtruth[1:]):

        new_x, new_y, new_z = prev_pose[0:3, 3]
        ax.scatter(new_x, new_z, color='g', marker='.')
        prev_pose = np.matmul(prev_pose, pose)

    new_x, new_y, new_z = prev_pose[0:3, 3]
    ax.scatter(new_x, new_z, color='g')
    plt.savefig(directory + '.png')


def generate_trajectory(gtruth_xyz, gtruth_rxyz, directory):

    poses_gtruth = []
    poses_pred = []
    print(gtruth_xyz.shape, gtruth_rxyz.shape)
    # exit(-1)
    for i, rotation in enumerate(gtruth_rxyz):
        curr_pose = np.zeros((4,4))
        r = R.from_quat(rotation)
        # print(i)
        curr_pose[0:3, 0:3] = r.as_dcm()
        curr_pose[0:3, 3] = gtruth_xyz[i]
        curr_pose[3, :] = [0, 0, 0, 1]
        poses_gtruth.append(curr_pose)
        # if i > 20:
        #     break

    plot_trajectory(poses_gtruth, directory)

def read_trajectory_files(fileset):

    files = np.sort(glob.glob(fileset + '/*.txt', recursive=True))
    gtruth_xyz = [[0, 0, 0]]
    gtruth_rxyz = [[0, 0, 0, 1]]
    for file in files:
        lines = open(file, 'r')
        lines = lines.readlines()
        line = lines[1]
        values = [float(v) for v in line.split()[1:]]
        gtruth_xyz.append(values[0:3])
        gtruth_rxyz.append(values[3:])
        # print(file, len(gtruth_xyz))
            # for value in words:
    return np.asarray(gtruth_xyz), np.asarray(gtruth_rxyz)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gtruth', type = str, default = '/cmlscratch/arjgpt27/projects/CMSC733/odom_data_rgb/ground_truth/ground_truth')
    parser.add_argument('--seq', type = str, default = '9')
    parser.add_argument('--pred', type = str, default = '/cmlscratch/arjgpt27/projects/CMSC733/output_pose_resnet')
    Flags = parser.parse_args()

    experiment_name = Flags.pred.split('/')[-1]
    Flags.seq = Flags.seq.zfill(2)
    directory = os.path.join('trajectory_plots', experiment_name, Flags.seq)
    if not os.path.exists(directory):
        os.makedirs(directory)

    Flags.gtruth = os.path.join(Flags.gtruth, Flags.seq)
    Flags.pred = os.path.join(Flags.pred, Flags.seq)

    gtruth_xyz, gtruth_rxyz = read_trajectory_files(Flags.gtruth)
    pred_xyz, pred_rxyz = read_trajectory_files(Flags.pred)

    pred_rxyz = pred_rxyz[0:gtruth_xyz.shape[0]]
    pred_xyz = pred_xyz[0:gtruth_xyz.shape[0]]
    offset = gtruth_xyz - pred_xyz
    pred_xyz += offset
    scale = np.sum(gtruth_xyz * pred_xyz, axis=1)/np.sum(pred_xyz ** 2 + 1e-08, axis=1)
    pred_xyz = pred_xyz*scale[:, np.newaxis]

    generate_trajectory(gtruth_xyz, gtruth_rxyz, directory + '/gtruth')
    generate_trajectory(pred_xyz, pred_rxyz, directory + '/pred')
