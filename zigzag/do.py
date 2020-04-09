import os
import tensorflow as tf

from zigzag import zigzag
from model_util import *
from util import dict_load,dict_save,read_train_valid_data,read_test_data,get_newest,iflarger,ifsmaller,saveAsNiiGz
from arg_parser import args_process
from process import train_valid_generator,test_generator,recover

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

ret_dict = args_process()

input_shape_dict = {'HaN_OAR': (224, 224), 'Thoracic_OAR': (256, 256), 'Naso_GTV': (224, 224), 'Lung_GTV': (256, 256)}
crop_range_dict = {'HaN_OAR': {'x': (160, 384), 'y': (160, 384)},
                   'Thoracic_OAR': {'x': (117, 417), 'y': (99, 399)},
                   'Lung_GTV': {'x': (117, 417), 'y': (99, 399)},
                   'Naso_GTV': {'x': (160, 384), 'y': (160, 384)}}
num_class_dict = {'HaN_OAR': 23, 'Thoracic_OAR': 7, 'Lung_GTV': 2, 'Naso_GTV': 2}

if(ret_dict['task'] == 'train'):

    # target selection
    target = ret_dict['target']

    # rarely changing options
    input_shape = input_shape_dict[target]
    crop_x_range = crop_range_dict[target]['x']
    crop_y_range = crop_range_dict[target]['y']
    resize_shape = (512, 512)
    num_class = num_class_dict[target]
    initial_channel = 64
    max_epoches = 200

    # usually changing options
    last = ret_dict['last']
    start_epoch = ret_dict['start_epoch']
    pattern = ret_dict['model_pattern']
    model_key = ret_dict['model']
    batch_size = 2
    learning_rate = 0.0001
    keep_prob = 0.5

    base_path = 'build/{}-{}'.format(model_key, target)
    pb_path = "build/{}-{}/frozen_model".format(model_key, target)
    ckpt_path = "build/{}-{}/ckpt".format(model_key, target)
    valid_log_metric_only_path = 'build/{}-{}/valid_metric_loss_only.log'.format(model_key, target)
    detail_log_path = "build/{}-{}/valid_detail.log".format(model_key, target)

    train_path_list,valid_path_list = read_train_valid_data("dataset/{}_train_dataset.txt".format(target), valid_rate=0.3, ifrandom=True)
    one_epoch_steps = len(train_path_list)//batch_size
    decay_patientce = 3
    valid_step = 5

    # input_shape is the shape of numpy array, but it isn't same as the opencv.
    # I hate opencv.
    train_batch_generator = train_valid_generator(train_path_list, len(train_path_list), True, batch_size, num_class,\
                                            input_shape, resize_shape, crop_x_range, crop_y_range, True, True, 15)

    valid_batch_generator = train_valid_generator(valid_path_list, len(valid_path_list), True, batch_size, num_class, \
                                            input_shape, resize_shape, crop_x_range, crop_y_range, False, False, 0)

    net = zigzag(base_path)
    graph = net.graph_compose(num_class, initial_channel)
    loss = net.loss_compose()
    metric = net.metric_compose()
    optimizer,lr,lr_init_op,lr_decay_op = net.optimizer_compose()

    with graph.as_default():
        init = tf.global_variables_initializer()
        sess = tf.Session(config=config, graph=graph)
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(scope='network'))
        if(last):
            net.restore(pattern, sess, graph, saver)
        # 记录训练过程的字符串
        show_string = None
        # 用于比较是否进行学习率衰减的
        saved_valid_log_epochwise = {'loss': [100000], 'metric': [0]}
        # 学习率衰减标志位
        learning_rate_descent_flag = 0
        # 将学习率初始化
        sess.run(lr_init_op, feed_dict={"lr_input:0": learning_rate})
        # 开始训练前，检查一遍权重保存路径
        if(not os.path.exists(ckpt_path)):
            os.makedirs(ckpt_path, 0o777)
        if(not os.path.exists(pb_path)):
            os.makedirs(pb_path, 0o777)
        if(not os.path.exists(valid_log_metric_only_path)):
            valid_log_dict = {}
            valid_log_dict['stepwise'] = {'loss':{}, 'metric':{}}
            valid_log_dict['epochwise'] = {'loss':[], 'metric':[]}
        else:
            valid_log_dict = dict_load(valid_log_metric_only_path)
        for i in range(start_epoch-1, max_epoches):
            # one epoch
            # 打开 log 文件
            detail_log = open(detail_log_path, 'a')
            # 不同 epoch 的训练中间结果分别保存
            valid_log_dict['stepwise']['loss'][i+1] = []
            valid_log_dict['stepwise']['metric'][i+1] = []
            # 用来学习率衰减的指标
            one_epoch_avg_loss = 0
            one_epoch_avg_metric = 0
            epochwise_train_generator = train_batch_generator.epochwise_iter()
            epochwise_valid_generator = valid_batch_generator.epochwise_iter()
            show_string = "epoch {}\ntrain dataset number:{} valid dataset number:{}"\
                            .format(i+1, train_batch_generator.slice_count, valid_batch_generator.slice_count)
            print(show_string)
            detail_log.write(show_string + '\n')
            for j in range(one_epoch_steps):
                # one step
                data,label = next(epochwise_train_generator)
                feed_dict = {'data:0': data, 'label:0': label, 'keep_probability:0': keep_prob}
                _ = sess.run(optimizer, feed_dict=feed_dict)
                if((j+1)%valid_step == 0):
                    data,label = next(epochwise_valid_generator)
                    feed_dict = {'data:0': data, 'label:0': label, 'keep_probability:0': keep_prob}
                    loss_key = list(loss.keys())
                    loss_show_string = ''
                    for key in loss_key:
                        if(key == 'loss'): continue
                        los = sess.run(loss[key], feed_dict=feed_dict)
                        loss_show_string += '{}:{} '.format(key, los)
                    metric_key = list(metric.keys())
                    metric_show_string = ''
                    for key in metric_key:
                        if(key == 'metric'): continue
                        met = sess.run(metric[key], feed_dict=feed_dict)
                        metric_show_string += '{}:{} '.format(key, met)
                    los,met = sess.run([loss['loss'], metric['metric']], feed_dict=feed_dict)
                    # 保存每次 valid 的指标
                    valid_log_dict['stepwise']["loss"][i+1].append(los)
                    valid_log_dict['stepwise']["metric"][i+1].append(met)
                    one_epoch_avg_loss += los
                    one_epoch_avg_metric += met
                    show_string = "epoch:{} steps:{}/{} valid_loss:{} valid_dice:{} learning_rate:{} {} {}"\
                                    .format(i+1, j+1, one_epoch_steps, los, met, learning_rate, loss_show_string, metric_show_string)
                    print(show_string)
                    detail_log.write(show_string + '\n')

            one_epoch_avg_loss = one_epoch_avg_loss/(one_epoch_steps//valid_step)
            one_epoch_avg_metric = one_epoch_avg_metric/(one_epoch_steps//valid_step)
            show_string = "=======================================================\n\
epoch_end epoch:{} epoch_avg_loss:{} epoch_avg_metric:{}\n"\
            .format(i+1, one_epoch_avg_loss, one_epoch_avg_metric)

            # 以 metric 为基准，作为学习率衰减的参考指标
            if(not iflarger(saved_valid_log_epochwise["metric"], one_epoch_avg_metric)):
                learning_rate_descent_flag += 1
            
            show_string += "learning_rate_descent_flag:{}\n".format(learning_rate_descent_flag)
            
            if(learning_rate_descent_flag == decay_patientce):
                learning_rate_once = learning_rate
                _ = sess.run(lr_decay_op)
                learning_rate = sess.run(lr)
                show_string += "learning rate decay from {} to {}\n".format(learning_rate_once, learning_rate)
                learning_rate_descent_flag = 0

            if(iflarger(saved_valid_log_epochwise["metric"], one_epoch_avg_metric)):
                show_string += "ckpt_model_save because of {}<={}\n"\
                                .format(saved_valid_log_epochwise["metric"][-1], one_epoch_avg_metric)
                show_string += net.save(sess, saver, i+1, pattern, one_epoch_avg_metric)
                saved_valid_log_epochwise['metric'].append(one_epoch_avg_metric)
                learning_rate_descent_flag = 0

            if(ifsmaller(saved_valid_log_epochwise["loss"], one_epoch_avg_loss)):
                saved_valid_log_epochwise['loss'].append(one_epoch_avg_loss)

            # 保存每个 epoch 的平均指标
            valid_log_dict['epochwise']['loss'].append(one_epoch_avg_loss)
            valid_log_dict['epochwise']['metric'].append(one_epoch_avg_metric)
            
            # 保存 log_dict
            dict_save(valid_log_dict, valid_log_metric_only_path)

            show_string += "======================================================="
            print(show_string)
            detail_log.write(show_string + '\n')
        saver.save(sess, "{}/best_model".format(ckpt_path))
        frozen_graph(sess,"{}/last.pb".format(pb_path))
        sess.close()

elif(ret_dict['task'] == 'test'):

    # target selection
    target = ret_dict['target']

    # rarely changing options
    input_shape = input_shape_dict[target]
    crop_x_range = crop_range_dict[target]['x']
    crop_y_range = crop_range_dict[target]['y']
    resize_shape = (512, 512)
    num_class = num_class_dict[target]

    # usually changing options
    keep_prob = 1
    batch_size = 2
    model_key = ret_dict['model']

    frozen_model_path = "build/{}-{}/frozen_model".format(model_key, target)

    frozen_model_name = os.path.basename(get_newest(frozen_model_path))

    graph = load_graph(get_newest(frozen_model_path))

    test_path_list = read_test_data("dataset/{}_test_dataset.txt".format(target))

    test_batch_generator = test_generator(test_path_list, batch_size, num_class, input_shape, resize_shape, \
                                          crop_x_range, crop_y_range)
    
    test_result_root_path = 'build/{}-{}/test_result/{}'.format(model_key, target, frozen_model_name)
    
    if(not os.path.exists(test_result_root_path)):
        os.makedirs(test_result_root_path, 0x777)

    patientwise_test_generator = test_batch_generator.patientwise_iter()
    with graph.as_default() as g:
        # loss & metric & optimizer relating things.
        predict = g.get_tensor_by_name('predict:0')
        softmax = tf.nn.softmax(predict)
        argmax = tf.argmax(softmax, axis=-1)
        sess = tf.Session(graph=g, config=config)
        for patient_name,nii_meta_info,single_patient_generator in patientwise_test_generator:
                # 单个病人
                patient_test_argmax = []
                patient_test_label = []
                for data,label in single_patient_generator:
                    feed_dict = {'data:0': data, 'keep_probability:0': keep_prob}
                    argmax = sess.run(argmax, feed_dict)
                    for i in range(argmax.shape[0]):
                        patient_test_argmax.append(argmax[i])
                        patient_test_label.append(label[i])
                patient_test_label = np.array(patient_test_label)
                patient_test_argmax_recover = recover(patient_test_argmax, patient_test_label.shape[:3], \
                                                      ifprocess, num_class, crop_x_range, crop_y_range, resize_shape)
                patient_test_argmax_recover = patient_test_argmax_recover.astype(np.uint8)
                test_result_save_dir_path = '{}/{}'.format(test_result_root_path, patient_name)
                if(not os.path.exists(test_result_save_dir_path)):
                    os.makedirs(test_result_save_dir_path, 0x777)
                test_result_save_path = test_result_save_dir_path + '/predict.nii.gz'
                saveAsNiiGz(predict_argmax_recover, save_path, info_dict['spacing'], info_dict['origin'])

else:
    print('Sorry,{} isn’t a valid option'.format(sys.argv[1]))
    exit()

