# coding = utf-8
import argparse
import collections
import itertools
import os
import sys

sys.path.append("/opt/develop/workspace/sohu/news/reinforce_learning/reinforce_train_15m")
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

import Console
from ddqn import ddqn_model_10_batch2 as ddqn
from environment import environment_10_newsTags_batch as env
from utility import replay
from utility.hype_parameters import HypeParameters as hypes

Transition = collections.namedtuple("Transition",
                                    ["state", "action", "reward", "next_state", "done", "max_news_num", "loadid"])
MAX_DATA_FILE = 48
MAX_LOG_FILE = 96


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # learning rate
    parser.add_argument('--policy_lr_rate', type=float,
                        help='policy net learning rate.',
                        default=0.0001)
    parser.add_argument('--optimizer', type=str,
                        help='optimizer.',
                        default='ADAM')

    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='saved_models/ckpt/online')

    parser.add_argument('--saved_model_dir', type=str,
                        help='dir for saving model.',
                        default='saved_models')

    parser.add_argument('--logs_dir', type=str,
                        help='dir for log.',
                        default='logs')

    parser.add_argument('--buffer_size', type=int,
                        help='max buffer_size for start of training',
                        default=200000)

    parser.add_argument('--batch_size', type=int,
                        help='batch_size for training',
                        default=64)

    parser.add_argument('--average_size', type=int,
                        help='size for time estimation of test',
                        default=100)

    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='gpu',
                        default=0.4)

    parser.add_argument('--cliped_norm', type=float,
                        help='',
                        default=500)

    parser.add_argument('--tau', type=float,
                        help='',
                        default=0.001)

    parser.add_argument('--gpuID', type=str,
                        help='gpu ID',
                        default='0')

    parser.add_argument('--number_epoch', type=int,
                        help='',
                        default=1)

    parser.add_argument('--train_data', type=str,
                        help='newsID 2 newsTags ',
                        default='data')

    return parser.parse_args(argv)


def get_model_info(path):
    file = open(path, 'r')
    result = dict()
    lines = file.read().strip().split("\n")
    for line in lines:
        tmp = line.split("=")
        result[tmp[0]] = tmp[1]
    file.close()
    return result


def write_model_infor(dic, path):
    file = open(path, "w")
    for key in dic.keys():
        line = key + "=" + dic[key] + "\n"
        file.write(line)
    file.close()


def write_episode_reward_infor(dic, path):
    file = open(path, "a")
    for key in dic.keys():
        line = key + "=" + dic[key] + "\n"
        file.write(line)
    file.close()


def get_model_filename(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    # elif len(meta_files) > 1:
    #     raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[-1]
    p = meta_file.rfind('.')
    model_file = model_dir + '/' + meta_file[0:p]
    return model_file


def train(sess, userEnv, actor_Q, saver, summary_writer, checkpoint_path, args, replay_buffer):
    epoch = 0
    global_step = -1
    uid_count = 0
    add_step = 0

    episode_reward_total = 0.
    episode_count = 0

    start_time = time.time()

    while epoch < args.number_epoch:
        state = userEnv.reset()
        uid_count += 1
        if userEnv.one_epoch_done:
            epoch += 1
            uid_count = 0

        if len(userEnv.episode_record) == 0:
            continue  # number of valid data is 0
        if userEnv.episode_record[0].current_window == "false":
            print('episode_record[0].current_window == false')
            continue
        '''
        while uid_count < 100:
            state=userEnv.reset()
            uid_count+=1
        '''
        # one step
        episode_reward = 0

        current_time = time.time()
        # 最多训练28分钟
        if current_time - start_time >= 28 * 60:
            break

        for t in itertools.count():
            if t >= len(userEnv.episode_record):
                break
            start_time = time.time()
            # read action from user record
            picked_newsIDs = userEnv.episode_record[t].recommend_newsIDs
            # print(userEnv.episode_record[t].userID)
            action = np.zeros(shape=[500, 10])

            for j in range(len(picked_newsIDs)):
                for i in range(len(userEnv.episode_record[t].indexSeq)):
                    if picked_newsIDs[j].__eq__(userEnv.episode_record[t].indexSeq[i]):
                        action[i][j] = 1.0
                        break

            next_state, reward, done, max_news_num, loadid = userEnv.step(action, t)
            episode_reward = pow(hypes.gamma, t) * reward

            # Keep track of the transition

            replay_buffer.add(
                Transition(state=state, action=action, reward=reward, next_state=next_state, done=done,
                           max_news_num=max_news_num, loadid=loadid))
            add_step += 1
            # Keep adding experience to the memory until
            # there are at least minibatch size samples

            if (
                    replay_buffer.size() >= args.batch_size and add_step % args.batch_size == 0) or replay_buffer.size() >= 100000:
                print("size =", replay_buffer.size())
                s_batch, a_batch, r_batch, s2_batch, t_batch, max_news, loadid = replay_buffer.sample_batch(
                    args.batch_size)
                print(loadid)
                y_batch = []

                target_q_batch = actor_Q.predict_target_value2(s2_batch)

                action_predict, target_q = get_max_reword_action(target_q_batch, t_batch, max_news)

                for i in range(args.batch_size):
                    if t_batch[i]:
                        y_batch.append(r_batch[i])
                    else:
                        y = r_batch[i] + actor_Q.gamma * target_q[i]
                        y_batch.append(y)

                _train_op, _loss, _value_estimate, _merge, _y, _candi_user_nfm, _grads, _q_value = actor_Q.train(
                    s_batch, a_batch,
                    y_batch)
                print(_q_value)
                print(_value_estimate)
                print(y_batch)
                actor_Q.update_target_network()
                global_step += 1

                duration = time.time() - start_time

                target_ave = 0
                q_value_ave = 0
                tq_c = 0
                for i, s in enumerate(s_batch):
                    if s_batch[i].position == 1:
                        target_ave += y_batch[i]
                        q_value_ave += _value_estimate[i]
                        tq_c += 1
                if tq_c > 0:
                    target_ave = target_ave / tq_c
                    q_value_ave = q_value_ave / tq_c
                else:
                    target_ave = 0.
                    q_value_ave = 0.

                summary_tq = tf.Summary()
                summary_tq.value.add(tag='target_ave', simple_value=target_ave)
                summary_tq.value.add(tag='q_value_ave', simple_value=q_value_ave)
                summary_writer.add_summary(summary_tq, global_step)

                print('epoch:', epoch,
                      '\tstep:', global_step,
                      '\tadd_step', add_step,
                      '\ttarget:%.3f' % target_ave,
                      '\tpredicted_q:%.3f' % q_value_ave,
                      '\tloss_critic:%.3f' % _loss,
                      '\ttime:%.3f s' % duration)

            if global_step % 200 == 0:
                saver.save(sess, checkpoint_path)

            if done:
                summary = tf.Summary()
                summary.value.add(tag='episode_reward', simple_value=episode_reward)
                episode_reward_total += episode_reward
                episode_count += 1
                summary_writer.add_summary(summary, global_step)
                break
            state = next_state

    if episode_count > 10:
        episode_reward_ave = episode_reward_total / episode_count
    else:
        episode_reward_ave = 0
        print("episode_count<10")
    return episode_reward_ave


def get_max_reword_action(target_q_batch, t_batch, max_news):
    shape = np.shape(target_q_batch)
    actions = []
    q_next_target = []
    for i in range(shape[0]):
        if t_batch[i]:
            action_i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            q_news_i = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            action_i, q_news_i = get_actions(target_q_batch[i], max_news[i])
        actions.append(action_i)
        q_next_target.append(sum(q_news_i))
    return actions, q_next_target


def find_max(seq):
    index = 0
    value = seq[index]
    for i in range(len(seq)):
        if seq[i] > value:
            index = i
            value = seq[index]
    return index, value


def get_actions(q_next_i, max_news_num):
    q_result = [-1] * 10
    action = [-1] * 10
    for j in range(10):
        qj = q_next_i[:max_news_num, j]
        index_j = -1
        value_j = -1e12
        while action.__contains__(index_j):
            index_j, value_j = find_max(qj)
            qj[index_j] = -1e12
        if index_j == -1:
            print("#### line 120 dqn\t", qj)
        action[j] = index_j
        q_result[j] = value_j
    return action, q_result


def evaluation(sess, userEnv, actor, time_size):
    epoch = 0
    test_time = 0
    global_step = 0
    uid_count = 0

    while epoch < hypes.number_epoch and global_step < time_size:
        state = userEnv.reset()
        uid_count += 1
        print('epoch: ', epoch, '\tuid_count: ', uid_count)
        '''
        while uid_count < 100:
            state=userEnv.reset()
            uid_count+=1
        '''
        if len(userEnv.episode_record) == 0:
            print("empty episode!")
            continue  # number of valid data is 0

        # one step
        for t in itertools.count():
            # read action from user record
            picked_newsIDs = userEnv.episode_record[t].recommend_newsIDs
            valid = True
            action = []
            for id in picked_newsIDs:
                if (id not in userEnv.episode_record[t].indexSeq):
                    valid = False
                    break
                else:
                    a = userEnv.episode_record[t].indexSeq.index(id)
                    action.append(a)

            if valid == False:
                print('the episode is not valid,')
                break

            next_state, reward, done = userEnv.step(action, t)

            t_s = time.time()
            target_action, target_probs, value_estimate = actor.predict_target(state)
            t_e = time.time() - t_s
            if global_step >= 5:
                test_time += t_e
            global_step += 1

            probs_str = ''
            for p in target_probs:
                probs_str += '{:.4f}'.format(p) + ','

            print('action_time:%.3f' % t_e,
                  '\tvalue_estimate:%.3f:' % value_estimate,
                  '\ttarget_action:', target_action)

            if (global_step == time_size):
                print('global_step:', global_step,
                      '\ttime average:', test_time / (global_step - 5))
                break
            if done:
                break
            state = next_state
        if userEnv.one_epoch_done:
            epoch += 1
            uid_count = 0


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=None)
    return output_graph_def


def saveAsPb(sess, path):
    gd = sess.graph.as_graph_def()
    for node in gd.node:

        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, gd,
        output_node_names=["train/Actor/candidates_input",
                           "train/Actor/candidates_mask",
                           "train/Actor/user_input",
                           "train/Actor/user_input_w",
                           "train/Actor/history_newsTags", "train/Actor/history_clicks",
                           "train/Actor/history_positions", "train/Actor/target_Q_net/q_out/clip_by_norm"],
        variable_names_whitelist=None)
    tf.train.write_graph(output_graph_def, path, "graph.pb", as_text=False)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID  # GPU

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction, allow_growth=False)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=False))
    # print("allocate gpu")
    with tf.variable_scope("train") as scope:
        actor_Q = ddqn.ActorPickNewsQ(learning_rate=args.policy_lr_rate,
                                      tau=args.tau,
                                      gamma=hypes.gamma,
                                      batch_size=args.batch_size,
                                      num_pre_vars=0,
                                      optimizer=args.optimizer,
                                      max_candidates_num=hypes.max_candidates_num,
                                      max_userTags_num=hypes.max_userTags_num,
                                      cliped_norm=args.cliped_norm)

        scope.reuse_variables()
    # Create a saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=2)

    model_save_path = args.saved_model_dir
    current_ckpt_path = model_save_path + "/ckpt/online/model.ckpt"
    print("current_ckpt_path:", current_ckpt_path)
    current_pb_path = model_save_path + "/pb/online/"
    model_property_path = model_save_path + "/properties"
    model_property = get_model_info(model_property_path)

    train_data = args.train_data

    logs_dir = args.logs_dir
    # summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

    # replay_buffer定义在循环外部，可以融合多个时间段的数据
    replay_buffer = replay.ReplayBuffer(args.buffer_size, hypes.random_seed)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        if args.pretrained_model != '' and os.path.exists(args.pretrained_model + '/checkpoint') == True:
            print('restore form:', current_ckpt_path)
            saver.restore(sess, current_ckpt_path)

        while True:
            old_model = model_property["model_time"]
            files = Console.getDirFiles(train_data)
            train_files = Console.getTrainFiles(files, old_model)
            print("train_files:", train_files)

            if len(train_files) < 1:
                time.sleep(10 * 60)
            else:
                for file_num in range(len(train_files)):
                    train_file_tmp = train_files[file_num]
                    # old_model = train_file_tmp.split("/")[-1]
                    _, old_model = os.path.split(train_file_tmp)
                    print("#### line 401\t", train_file_tmp, "\t", old_model)
                    model_timeline_ckpt = model_save_path + "/ckpt/timeline/" + old_model + "/model.ckpt"
                    model_timeline_pb = model_save_path + "/pb/timeline/" + old_model

                    newsTag_dict_path = train_file_tmp + "/newsOnehot"
                    user_records_path = train_file_tmp + "/userOnehot"

                    if os.path.exists(newsTag_dict_path) == False or os.path.exists(user_records_path) == False:
                        print('There is no data file')
                        continue

                    data_path = env.DataPaths(newsTag_dict_path, user_records_path)
                    userEnv = env.Environment(data_path, max_newsTag_num=hypes.max_newsTag_num,
                                              max_userTags_num=hypes.max_userTags_num,
                                              max_candidates_num=hypes.max_candidates_num)

                    print('traing....')
                    # now = time.strftime('%Y-%m-%d_%H_%M')
                    now_log_dir = os.path.join(logs_dir, old_model)
                    summary_writer = tf.summary.FileWriter(now_log_dir, sess.graph)

                    episode_reward_ave = train(sess, userEnv, actor_Q, saver, summary_writer, current_ckpt_path, args,
                                               replay_buffer)
                    if episode_reward_ave == 0:
                        continue

                    print('write_episode_reward_infor:', old_model)
                    episode_reward_info = {}
                    episode_reward_info[old_model] = "{0:.4f}".format(episode_reward_ave)
                    write_episode_reward_infor(episode_reward_info, model_save_path + "/episode_rewards.txt")

                    print('save model:', current_ckpt_path)
                    saver.save(sess, current_ckpt_path)
                    print('save model:', model_timeline_ckpt)
                    saver.save(sess, model_timeline_ckpt)

                    print('save pb:', current_pb_path)
                    saveAsPb(sess, current_pb_path)
                    print('save pb:', model_timeline_pb)
                    saveAsPb(sess, model_timeline_pb)

                    print('updateVideoChannelModel....')

                    # online
                    # shell_path = "/data/bd-recommend/songfeng/project/transfer_log/updateVideoChannelModel.sh"
                    # os.system(shell_path)

                    # dev
                    # shell_path = "/data/bd-recommend/lizhenmao/transfer_log/updateVideoChannelModel.sh"
                    # os.system(shell_path)

                    # print('testing....')
                    # evaluation(sess, userEnv, actor_Q, args.average_size)

                    model_property["model_time"] = train_file_tmp.split("/")[-1]
                    print('write_model_property:', model_property["model_time"])
                    write_model_infor(model_property, model_property_path)

                # online 注释去掉
                # Console.delete_datas(files, MAX_DATA_FILE)

                timeline_ckpt_files = Console.getDirFiles(model_save_path + "/ckpt/timeline/")
                timeline_pb_files = Console.getDirFiles(model_save_path + "/pb/timeline/")
                Console.delete_datas(timeline_ckpt_files, MAX_DATA_FILE)
                Console.delete_datas(timeline_pb_files, MAX_DATA_FILE)

                timeline_log_files = Console.getDirFiles(logs_dir)  # 删除log文件
                Console.delete_datas(timeline_log_files, MAX_LOG_FILE)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
