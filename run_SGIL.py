import tensorflow as tf
import numpy as np
import os, pdb, sys
from time import time
from tqdm import tqdm
from shutil import copyfile
import argparse
sys.path.append('./')
from utils.rec_dataset import Dataset
from utils.log import Logger
from utils.evaluate import *
from models.SGIL import SGIL
np.random.seed(2024)
tf.set_random_seed(2024)


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Parameters')
    parser.add_argument('--dataset', type=str, default='douban_book', help='?')
    parser.add_argument('--runid', type=str, default='0', help='current log id')
    parser.add_argument('--device_id', type=str, default='0', help='?')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--topk', type=int, default=20, help='Topk value for evaluation')   # NDCG@20 as convergency metric
    parser.add_argument('--early_stops', type=int, default=10, help='model convergent when NDCG@20 not increase for x epochs')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negetiva samples for each [u,i] pair')
    parser.add_argument('--social_noise_ratio', type=float, default=0, help='?')

    ### model parameters ###
    parser.add_argument('--gcn_layer', type=int, default=3, help='?')
    parser.add_argument('--num_user', type=int, default=13024, help='max uid')
    parser.add_argument('--num_item', type=int, default=22347, help='max iid')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent embedding dimension')
    parser.add_argument('--l2_reg', type=float, default=2e-4, help='?')
    parser.add_argument('--edge_bias', type=float, default=0.5, help='observation bias of social relations')
    parser.add_argument('--num_envs', type=int, default=4, help='')
    parser.add_argument('--penalty_coff', type=float, default=0.10, help='')
    parser.add_argument('--adv_bs', type=int, default=5, help="?")
    return parser.parse_args()


def makir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_file(save_path):
    copyfile('./models/PairWise_model.py', save_path + 'PariWise_model.py')
    copyfile('./models/SGIL.py', save_path + 'SGIL.py')
    copyfile('./run_SGIL.py', save_path + 'run_SGIL.py')
    copyfile('./utils/rec_dataset.py', save_path + 'rec_dataset.py')
    copyfile('./utils/evaluate.py', save_path + 'evaluate.py')



if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'yelp':
        args.num_user = 19539
        args.num_item = 22228
        args.lr = 0.001
        args.batch_size = 2048
    elif args.dataset == 'epinions':
        args.num_user = 18202
        args.num_item = 47449
        args.lr = 0.001
        args.batch_size = 2048

    args.data_path = './datasets/' + args.dataset + '/'
    record_path = './saved/' + args.dataset + '/' + args.runid + '/'
    model_save_path = record_path + 'models/'
    makir_dir(model_save_path)
    save_file(record_path)
    log = Logger(record_path)
    for arg in vars(args):
        log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    rec_data = Dataset(args)
    rec_model = SGIL(args, rec_data)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)

    # *****************************  start training  *******************************#
    writer = tf.summary.FileWriter(record_path + '/log/', sess.graph)
    max_hr, max_recall, max_ndcg, early_stop = 0, 0, 0, 0
    topk = args.topk
    ndcg_list = []
    for epoch in range(args.epochs):
        t1 = time()
        data_iter = rec_data._batch_sampling_softmax()
        sum_penalty, sum_loss1, sum_loss2, sum_loss3, batch_num = 0, 0, 0, 0, 0
        for batch_u, batch_i in tqdm(data_iter):
            feed_dict = {rec_model.users: batch_u, rec_model.pos_items: batch_i}
            _penalty, _loss1, _ = sess.run([rec_model.penalty, rec_model.loss, rec_model.opt1], feed_dict=feed_dict)
            if batch_num % args.adv_bs == 0:
                _exploration, _ = sess.run([rec_model.penalty, rec_model.opt2], feed_dict=feed_dict)
            sum_penalty += _penalty
            sum_loss1 += _loss1
            batch_num += 1
        mean_penalty = sum_penalty / batch_num
        mean_recloss = sum_loss1 / batch_num
        log.write('Epoch:{:d}, Loss_rec:{:.4f}, Loss_penalty:{:.4f}\n'.format(
            epoch, mean_recloss, mean_penalty))
        t2 = time()

        # ***************************  Evaluation on Top-20  *****************************#
        if (epoch % 1) == 0:
            early_stop += 1
            user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])
            hr, recall, ndcg = num_faiss_evaluate(rec_data.valdata, rec_data.traindata,
                                                  [topk], user_matrix, item_matrix,
                                                  rec_data.valdata.keys())  ### all users evaluation
            log.write('Epoch:{:d}, topk:{:d}, R@20:{:.4f}, P@20:{:.4f}, N@20:{:.4f}\n'.format(epoch, topk, recall[topk],
                                                                                              hr[topk], ndcg[topk]))
            ndcg_list.append(ndcg[topk])
            max_hr = max(max_hr, hr[topk])
            max_recall = max(max_recall, recall[topk])
            max_ndcg = max(max_ndcg, ndcg[topk])
            if ndcg[topk] == max_ndcg:
            # if recall[topk] == max_recall:
                early_stop = 0
                best_ckpt = 'epoch_' + str(epoch) + '_ndcg_' + str(ndcg[topk]) + '.ckpt'
                saver.save(sess, model_save_path + best_ckpt)
            t3 = time()
            log.write('traintime:{:.4f}, valtime:{:.4f}\n\n'.format(t2 - t1, t3 - t2))
            np.save(record_path + 'ndcg_list.npy', ndcg_list)
            if epoch > 30 and early_stop > args.early_stops:
                log.write('early stop\n')
                log.write('max_recall@20=:{:.4f}, max_ndcg@20=:{:.4f}\n'.format(max_recall, max_ndcg))
                np.save(record_path+'ndcg_list.npy', ndcg_list)
                break

    # ***********************************  start evaluate testdata   ********************************#
    writer.close()
    saver.restore(sess, model_save_path + best_ckpt)
    log.write('=================Validation results==================\n')
    user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb], feed_dict=feed_dict)
    hr, recall, ndcg = num_faiss_evaluate(rec_data.valdata, rec_data.traindata,
                                          [10, 20, 30, 40, 50], user_matrix, item_matrix,
                                          rec_data.valdata.keys())  ### all users evaluation
    for key in ndcg.keys():
        log.write('Topk:{:3d}, R@20:{:.4f}, P@20:{:.4f} N@20:{:.4f}\n'.format(key, recall[key], hr[key], ndcg[key]))

    log.write('=================Evaluation results==================\n')
    user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])
    hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [10, 20, 30, 40, 50], user_matrix, item_matrix,
                                          rec_data.testdata.keys())  ### all users evaluation
    for key in ndcg.keys():
        log.write('Topk:{:3d}, R@20:{:.4f}, P@20:{:.4f} N@20:{:.4f}\n'.format(key, recall[key], hr[key], ndcg[key]))
    log.close()