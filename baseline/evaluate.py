import os

from loss_metric import *
from arg_parser import arg_parser
from util import get_newest,readImage,one_hot,average

a = arg_parser()

metric_dict = {'Precision': Precision, 'IoU': IOU, 'DSC': DSC}
model_dict = {'unet': 'U-Net', 'r2u': 'R2U-Net', 'att': 'Attention U-Net', 'uplus': 'UNet++', 'ce': 'CE-Net', 'vnet': 'V-Net', \
              'zigzag_re': 'ZigZag-U-Net-residual', 'zigzag_re_r': 'ZigZag-U-Net-residual-reverse', 'zigzag_r': 'ZigZag-U-Net-regular', 'zigzag_r_r': 'ZigZag-U-Net-regular-reverse', \
              'zigzag_d': 'ZigZag-U-Net-dense', 'attv': 'Attention V-Net'}
target_dict = {'HN_OAR': 'HaN_OAR', 'Lu_OAR': 'Thoracic_OAR', 'HN_GTV': 'Naso_GTV', 'Lu_GTV': 'Lung_GTV'}
num_class_dict = {'HaN_OAR': 23, 'Thoracic_OAR': 7, 'Lung_GTV': 2, 'Naso_GTV': 2}

a.add_map('--model', model_dict)
a.add_map('--target', target_dict)

ret_dict = a()

model_key = ret_dict['--model']
target = ret_dict['--target']
num_class = num_class_dict[target]

label_root_path = os.path.join(r'E:\dataset\zhongshan_hospital\CSTRO\test', target)
test_list = os.listdir(label_root_path)
predict_root_path = get_newest(os.path.join('build', '{}-{}'.format(model_key, target), 'test_result'))
pb_name = os.path.split(predict_root_path)[1]
predict_process_root_path = os.path.join('build', '{}-{}'.format(model_key, target), 'test_result_process', pb_name)

def do(predict_rp, label_rp, metric_dict):
    metric_keys = list(metric_dict.keys())
    metric_list = {key:[] for key in metric_keys}
    metric_channelwise_list = {key:[] for key in metric_keys}
    f_dict = {key:open('{}/{}.txt'.format(predict_rp, key), 'w') for key in metric_keys}
    for key in metric_keys:
        f = f_dict[key]
        f.write('name ')
        for i in range(1, num_class):
            f.write('OAR%d '%(i))
        f.write('average\n')
    
    count = 0
    for patient in test_list:
        predict_path = os.path.join(predict_rp, patient, 'predict.nii.gz')
        label_path = os.path.join(label_rp, patient, 'label.nii.gz')
        
        predict = readImage(predict_path).astype(np.uint8)
        label = readImage(label_path)

        predict = one_hot(predict)
        label = one_hot(label)

        for key in metric_keys:
            f = f_dict[key]
            temp_list = []
            temp_string = 'patient{} '.format(patient)
            for i in range(1, num_class):
                metric = metric_dict[key](label[..., i], predict[..., i])
                temp_string += '%.4f '%(metric)
                temp_list.append(metric)
            avg = average(temp_list)
            temp_string += '%.4f\n'%(avg)
            metric_channelwise_list[key].append(temp_list)
            metric_list[key].append(avg)
            f.write(temp_string)
        
        count += 1
        print('patient{} done {}/{}'.format(patient, count, len(test_list)))

    for key in metric_keys:
        f = f_dict[key]
        temp_string = 'average '
        for i in range(1, num_class):
            temp_string += '%.4f '%(average([sub[i-1] for sub in metric_channelwise_list[key]]))
        temp_string += '%.4f\n'%(average(metric_list[key]))
        f.write(temp_string)
        f.close()

do(predict_root_path, label_root_path, metric_dict)
do(predict_process_root_path, label_root_path, metric_dict)