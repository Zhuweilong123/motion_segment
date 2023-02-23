import os
import numpy as np
import time
from dataset.data_prepocess.Matrix_Angle import R_to_angle
# transform poseGT [R|t] to [x, y, z, theta_x, theta_y, theta_z]
# 转换成欧拉角之前或之后还需要转个坐标系，因为打算使用到的color image是cam2下的，所以还需将cam0下的得到的pose进行一个平移计算
# save as .txt file
def create_pose_kitti():
	#读取的行数
	# info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}#这里删掉了序列10因为没有对应的color image
	#info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760],
			#'06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590]}
	info = {'10': [0, 1200]}
	start_t = time.time()
	for video in info.keys():
		file_name = '/home/zhanlei/sfmLearner/odometry_color_dataset/poses/{}.txt'.format(video)  # 这里的地址如果不一样的话，需要更改哦
		print('Transforming {} ...'.format(file_name))
		with open(file_name) as f:
			lines = [line.split('\n')[0] for line in f.readlines()]
			print(lines[0])
			t = Cam0T0Cam2(video)  # 每个序列计算得到的cam0→cam2对应的转换矩阵为t
			poses = [R_to_angle([float(value) for value in l.split(' ')], t) for l in lines]  # list of pose (pose=list of 12 floats)，这里传进参数t是让cam0坐标系先转换成cam2坐标系
			print(poses[0])
			poses = np.array(poses)
			file_name2 = '/home/zhanlei/sfmLearner/odometry_color_dataset/sequences/10/image_2_pose/{}.txt'.format(video)
			base_file_name = os.path.splitext(file_name2)[0]
			#print(poses[0])
			#np.save(base_file_name+'.npy', poses)
			np.savetxt(base_file_name+'.txt', poses)
			print('Video {}: shape={}'.format(video, poses.shape))
	print('elapsed time = {}'.format(time.time()-start_t))


def Cam0T0Cam2(seq):
	with open(os.path.join('/home/zhanlei/sfmLearner/odometry_color_dataset/sequences/{}/'.format(seq), 'calib.txt'),
			  'r') as f:  # os.path.join可以将多个路径拼接起来
		iPx = f.readlines()
	# homography matrix
	iP0 = np.array([float(i) for i in iPx[0].strip('\n').split(' ')[1:]]).reshape(3, 4)
	iP2 = np.array([float(i) for i in iPx[2].strip('\n').split(' ')[1:]]).reshape(3, 4)

	# instrincs of cams：相机的内参
	K0 = iP0[:3, :3]
	K2 = iP2[:3, :3]

	# calculate real translation
	K0_inv = np.linalg.inv(K0)  # np.linalg.inv矩阵求逆
	K2_inv = np.linalg.inv(K2)
	T0 = K0_inv @ (iP0[:3, 3].reshape(3, 1))  # @表示矩阵和数组的乘法，即使两杯的维度没有对齐，比如mxn@1xn=1xm，即第一个数组的第1行与第二个数组对应位置相乘相加后，得到最终结果的第1位，第一个数组的第2行与第二个数组对应位置相乘相加后，得到最终结果的第2位
	T2 = K2_inv @ (iP2[:3, 3].reshape(3, 1))
	# T3 = T2 - T0  # 这里只需要第一行因为yz轴都没有平移距离

	Trans = np.array([0, 0, 0, T2[0, 0] - T0[0, 0], 0, 0, 0, 0, 0, 0, 0, 0])
	Trans = Trans.reshape((3,4))

	return Trans

"""def create_pose_sevenscenes():
	info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
	start_t = time.time()
	for video in info.keys():
		file_name = 'KITTI/poses/{}.txt'.format(video) # 这里的地址如果不一样的话，需要更改哦
		print('Transforming {} ...'.format(file_name))
		with open(file_name) as f:
			lines = [line.split('\n')[0] for line in f.readlines()]
			#print(lines[0])
			poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines]  # list of pose (pose=list of 12 floats)
			print(poses[0])
			poses = np.array(poses)
			base_file_name = os.path.splitext(file_name)[0]
			#print(poses[0])
			np.save(base_file_name+'.npy', poses)
			print('Video {}: shape={}'.format(video, poses.shape))
	print('elapsed time = {}'.format(time.time()-start_t))
	"""

if __name__ == '__main__':
	#clean_unused_images()
	#选择是矩阵转欧拉角还是转四元数
	create_pose_kitti()
