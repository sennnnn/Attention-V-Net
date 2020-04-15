import os
import numpy as np

from arg_parser import arg_parser
from concurrent.futures import wait,ALL_COMPLETED,ThreadPoolExecutor
from util import one_hot,readNiiAll,saveAsNiiGz,get_newest
from process import lung_process

a = arg_parser()

model_dict = {'unet': 'U-Net', 'r2u': 'R2U-Net', 'att': 'Attention U-Net', 'uplus': 'UNet++', 'ce': 'CE-Net', 'vnet': 'V-Net', \
              'zigzag_re': 'ZigZag-U-Net-residual', 'zigzag_re_r': 'ZigZag-U-Net-residual-reverse', 'zigzag_r': 'ZigZag-U-Net-regular', 'zigzag_r_r': 'ZigZag-U-Net-regular-reverse', \
              'zigzag_d': 'ZigZag-U-Net-dense', 'attv': 'Attention V-Net'}

a.add_map('--model', model_dict)

ret_dict = a()

target = 'Thoracic_OAR'
model_key = ret_dict['--model']
num_class = 7

predict_root_path = get_newest(os.path.join('build', '{}-{}'.format(model_key, target), 'test_result'))
pb_name = os.path.split(predict_root_path)[1]
predict_process_root_path = os.path.join('build', '{}-{}'.format(model_key, target), 'test_result_process', pb_name)
label_root_path = os.path.join(r'E:\dataset\zhongshan_hospital\CSTRO\test', target)
test_list = os.listdir(label_root_path)

def son_process(left_lung_block, right_lung_block, index, patient, slice_count):
    left_lung = left_lung_block[index][117:417, 99:399]
    right_lung = right_lung_block[index][117:417, 99:399]
    lung_process(left_lung, right_lung)
    left_lung_block[index][117:417, 99:399] = left_lung
    right_lung_block[index][117:417, 99:399] = right_lung
    print('patient{} {}/{} done'.format(patient, index + 1, slice_count))

def main_process(predict_root_path, predict_process_root_path, num_class, patient):
    print('patient{} doing'.format(patient))
    predict_path = os.path.join(predict_root_path, patient, 'predict.nii.gz')
    spacing,origin,predict = readNiiAll(predict_path)
    predict_onehot = one_hot(predict, num_class).astype(np.uint8)
    
    left_lung_block = predict_onehot[:, :, :, 1]
    right_lung_block = predict_onehot[:, :, :, 2]

    slice_count = predict.shape[0]

    executor = ThreadPoolExecutor(max_workers=10)
    son_thread_task_list = []
    for index in range(slice_count):
        # son_process(left_lung_block, right_lung_block, index, patient, slice_count)
        son_thread_task_list.append(executor.submit(son_process, left_lung_block, right_lung_block, index, patient, slice_count))    

    wait(son_thread_task_list, return_when=ALL_COMPLETED)
    predict_onehot[:, :, :, 1] = left_lung_block 
    predict_onehot[:, :, :, 2] = right_lung_block 
    predict = np.argmax(predict_onehot, axis=-1).astype(np.uint8)

    save_path = os.path.join(predict_process_root_path, patient)
    if(not os.path.exists(save_path)):
        os.makedirs(save_path, 0o600)
    save_path = os.path.join(save_path, 'predict.nii.gz')
    saveAsNiiGz(predict, save_path, spacing, origin)

    print('patient{} done '.format(patient))

executor = ThreadPoolExecutor(max_workers=3)
thread_task_list = []
for patient in test_list:
    # main_process(predict_root_path, predict_process_root_path, num_class, patient)
    thread_task_list.append(executor.submit(main_process, predict_root_path, predict_process_root_path, num_class, patient))    

wait(thread_task_list, return_when=ALL_COMPLETED)

print('done!')


# # 左右肺排查，已经膨胀处理，这是针对肺部的操作
# # 测试出来的有效面积
# if(target == "Thoracic_OAR"):
#     if(findMaxContours(one_slice_mid_dst[..., 2]) > 300):
#         one_slice_mid_dst[..., 2] = left_lung_after_process(one_slice_mid_dst[...,2], one_slice_mid_dst[...,1])
#     if(findMaxContours(one_slice_mid_dst[..., 1]) > 300):
#         one_slice_mid_dst[..., 1] = right_lung_after_process(one_slice_mid_dst[...,1], one_slice_mid_dst[...,2])

