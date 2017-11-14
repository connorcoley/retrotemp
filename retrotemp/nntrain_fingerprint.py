import tensorflow as tf
from utils.nn import linearND
import math, sys, random, os
from optparse import OptionParser
import threading
from multiprocessing import Queue, Process
import numpy as np
from Queue import Empty
import time
import h5py
from itertools import chain
import os 
import cPickle as pickle
project_root = os.path.dirname(os.path.dirname(__file__))

NK = 100
NK0 = 10
report_interval = 10
max_save = 20
min_iterations = 1000

score_scale = 5.0
min_separation = 0.25

FP_len = 1024
FP_rad = 2

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path", default=os.path.join(project_root, 'data', 'reaxys_limit10.txt'))
parser.add_option("-m", "--save_dir", dest="save_path", default=os.path.join(project_root, 'models', 'example_model'))
parser.add_option("-b", "--batch", dest="batch_size", default=1024)
parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
parser.add_option("-o", "--out", dest="output_size", default=61142)
parser.add_option("-d", "--depth", dest="depth", default=5)
parser.add_option("-l", "--max_norm", dest="max_norm", default=5.0)
parser.add_option("-u", "--device", dest="device", default="")
parser.add_option("--test", dest="test", default='')
parser.add_option("-v", "--verbose", dest="verbose_test", default=False)
parser.add_option("-c", "--checkpoint", dest="checkpoint", default="final")
parser.add_option("-s", "--saveint", dest="save_interval", default=0)
parser.add_option("-i", "--interactive", dest="interactive", default=False)
opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
max_norm = float(opts.max_norm)
test = opts.test
save_interval = int(opts.save_interval)
verbose_test = bool(opts.verbose_test)
interactive_mode = bool(opts.interactive)
output_size = int(opts.output_size)

if interactive_mode:
    batch_size = 2 # keep it small

if not os.path.isdir(opts.save_path):
    os.mkdir(opts.save_path)

import rdkit.Chem.AllChem as AllChem
def mol_to_fp(mol, radius=FP_rad, nBits=FP_len):
    if mol is None:
        return np.zeros((nBits,), dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, 
        useChirality=True), dtype=np.bool)

def smi_to_fp(smi, radius=FP_rad, nBits=FP_len):
    if not smi:
        return np.zeros((nBits,), dtype=np.float32)
    return mol_to_fp(Chem.MolFromSmiles(smi), radius, nBits)

gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=opts.device)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    _input_mol = tf.placeholder(tf.float32, [batch_size, FP_len])
    _label = tf.placeholder(tf.int32, [batch_size,])

    q = tf.FIFOQueue(20, [tf.float32, tf.int32]) # fixed size
    enqueue = q.enqueue([_input_mol, _label])
    [input_mol, label] = q.dequeue()
    src_holder = [input_mol, label]

    input_mol.set_shape([batch_size, FP_len])
    label.set_shape([batch_size,])
    mol_hiddens = tf.nn.relu(linearND(input_mol, hidden_size, scope="encoder0"))
    for d in xrange(1, depth):
        mol_hiddens = tf.nn.relu(linearND(mol_hiddens, hidden_size, scope="encoder%i"%d))

    score = linearND(mol_hiddens, output_size, scope="output")
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=label))
    _, topk = tf.nn.top_k(score, k=NK)

    # For normal reaction-wise training
    _lr = tf.placeholder(tf.float32, [])
    optimizer = tf.train.AdamOptimizer(learning_rate=_lr)
    param_norm = tf.global_norm(tf.trainable_variables())
    grads_and_vars = optimizer.compute_gradients(loss / batch_size)
    grads, var = zip(*grads_and_vars)
    grad_norm = tf.global_norm(grads)
    new_grads, _ = tf.clip_by_global_norm(grads, max_norm)
    grads_and_vars = zip(new_grads, var)
    backprop = optimizer.apply_gradients(grads_and_vars)


    tf.global_variables_initializer().run(session=session)
    size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size_func(v) for v in tf.trainable_variables())
    print "Model size: %dK" % (n/1000,)

    queue = Queue()


    def read_data_once(path, coord, frag='valid'):
        print('Loading data file')
        with open(path + '.data_pkl', 'r') as f:
            data = pickle.load(f)
        print('Loading fingerprint file')
        with open(path + '.fp_pkl', 'r') as f:
            FPs = pickle.load(f)

        data_len = len(data)
        print('%i total data entries' % data_len)
        if frag == 'train':
            data = data[:int(0.8 * data_len)]
            FPs = FPs[:int(0.8 * data_len), :]
            data_len = len(data)
            print('Taking 0.8 as training set (%i)' % data_len)
        elif frag == 'valid':
            data = data[int(0.8 * data_len):int(0.9 * data_len)]
            FPs = FPs[int(0.8 * data_len):int(0.9 * data_len), :]
            data_len = len(data)
            print('Taking 0.1 as validation set (%i)' % data_len)
        elif frag == 'test':
            data = data[int(0.9 * data_len):]
            FPs = FPs[int(0.9 * data_len):, :]
            data_len = len(data)
            print('Taking 0.1 as test set (%i)' % data_len)
        else:
            raise ValueError('Unknown data frag type')
        it = 0
        src_mols = np.zeros((batch_size, FP_len), dtype=np.float32)
        src_labels = np.array([0 for i in range(batch_size)], dtype=np.int32)
        while it < data_len:

            # Try to get all FPs in one read (faster)
            if (it + batch_size) <= data_len:
                src_mols = FPs[it:it+batch_size, :].todense().astype(np.float32)
                src_labels = np.array([ex[4] for ex in data[it:it+batch_size]], dtype=np.int32) # template num
                src_info = data[it:it+batch_size]
                it = it + batch_size
            # If we are at the end, do one-by-one)
            else:
                src_info = []
                for i in xrange(batch_size):
                    if it >= data_len:
                        src_mols[i,:] = 0. # 0 out fingerprint
                        src_labels[i] = 0. # doesn't matter
                        src_info.append([])
                    else:
                        src_mols[i,:] = FPs[it,:].todense().astype(np.float32)
                        src_info.append(data[it])
                        src_labels[i] = data[it][4] # template_num
                    it = it + 1

            session.run(enqueue, feed_dict={_input_mol: src_mols, _label: src_labels})
            queue.put(src_info)
            # print('Queue size: {}'.format(queue.qsize()))
            # sys.stdout.flush()

        # Stop signal for testing
        queue.put(None)
        coord.request_stop()

    def read_data_master(path, coord):
        with open(path + '.data_pkl', 'r') as f:
            print('loading data')
            data = pickle.load(f)
                
        with open(path + '.fp_pkl', 'r') as f:
            print('loading sparse FP file')
            FPs = pickle.load(f)

        data_len = len(data)
        print('%i total data entries' % data_len)
        print('...slicing data')
        data = data[:int(0.8 * data_len)]  
        FPs = FPs[:int(0.8 * data_len), :]

        data_len = len(data)
        print('Taking 0.8 for training (%i)' % data_len)
        
        it = 0; 
        src_mols = np.zeros((batch_size, FP_len), dtype=np.float32)
        src_labels = np.array([0 for i in range(batch_size)], dtype=np.int32)
        while not coord.should_stop():

            # Try to get all FPs in one read (faster)
            # Try to get all FPs in one read (faster)
            if (it + batch_size) <= data_len:
                src_mols = FPs[it:it+batch_size, :].todense().astype(np.float32)
                src_labels = np.array([ex[4] for ex in data[it:it+batch_size]], dtype=np.int32) # template num
                src_info = data[it:it+batch_size]
                it = (it + batch_size) % data_len

            # If we are at the end (where we need to loop around, do one-by-one)
            else:
                src_info = []

                for i in xrange(batch_size):
                    src_mols[i,:] = FPs[it,:].todense().astype(np.float32)
                    src_info.append(data[it])
                    src_labels[i] = data[it][4] # template_num
                    it = (it + 1) % data_len
            # print(src_mols)
            # print(src_labels)
            session.run(enqueue, feed_dict={_input_mol: src_mols, _label: src_labels})
            queue.put(src_info)
            #print('Queue size: {}'.format(queue.qsize()))
            sys.stdout.flush()

        coord.request_stop()
        f.close()

    def dummy_thread():
        return

    coord = tf.train.Coordinator()
    if interactive_mode:
        all_threads = [threading.Thread(target=dummy_thread)]
    elif test:
        all_threads = [threading.Thread(target=read_data_once, args=(opts.train_path, coord), kwargs={'frag': opts.test})]
        print('Added read_data_once')
    else:
        all_threads = [threading.Thread(target=read_data_master, args=(opts.train_path, coord))]
        print('Added read_data_master')

    [t.start() for t in all_threads]

    if not interactive_mode:
        print('Reading data file to figugre out data length')
        with open(opts.train_path + '.data_pkl', 'r') as f:
            data_len = len(pickle.load(f))

        print('Data length: %i' % data_len)
        if save_interval == 0: # approx once per epoch
            save_interval = np.ceil(data_len / float(batch_size))

    saver = tf.train.Saver(max_to_keep=None)
    if test or interactive_mode:
        if opts.checkpoint:
            restore_path = os.path.join(opts.save_path, 'model.%s' % opts.checkpoint)
        else:
            restore_path = tf.train.latest_checkpoint(opts.save_path)
        saver.restore(session, restore_path)
        print('Restored values from latest saved file ({})'.format(restore_path))
        test_path = '%s.prediced.%s.%s' % (restore_path, os.path.basename(opts.train_path), str(opts.test))
        summary_path = os.path.join(opts.save_path, 'model.%s.summary' % os.path.basename(opts.train_path))
    it, sum_diff, sum_gnorm, = 0, 0.0, 0.0
    sum_loss = 0.0;
    sum_acc1 = 0.0;
    sum_acc5 = 0.0;
    sum_acc10 = 0.0;
    sum_acc20 = 0.0;
    sum_acc50 = 0.0;
    sum_acc100 = 0.0;

    lr = 0.001
    try:
        if interactive_mode:
            pass
            # prompt = raw_input('enter a tag for this session: ')
            # interactive_path = '%s.interactive.%s' % (restore_path, prompt.strip())
            # fid = open(interactive_path, 'a')

            # def get_score_from_smi(smi):
            #     if not smi:
            #         return ('', 0.)
            #     src_batch = [smi]
            #     while len(src_batch) != (batch_size * 2): # round out last batch
            #         src_batch.append('')
            #     src_mols = np.array(map(smi_to_fp, src_batch), dtype=np.float32)
            #     if sum(sum(src_mols)) == 0:
            #         print('Could not get fingerprint?')
            #         cur_score = [0.]
            #     else:
            #         # Run
            #         cur_score, = session.run([score], feed_dict={
            #             input_mol: src_mols,
            #             _lr: 0.001,
            #         })
            #         print('Score: {}'.format(cur_score[0]))
            #     mol = Chem.MolFromSmiles(smi)
            #     if mol:
            #         smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
            #     else:
            #         smi = ''
            #     return (smi, cur_score[0])

            # while True:
            #     try:
            #         prompt = raw_input('\nEnter SMILES (or quit): ')
            #         if prompt.strip() == 'quit':
            #             break
            #         if str('>>') in prompt: # reaction
            #             reactants = prompt.strip().split('>>')[0].split('.')
            #             reactants_smi = []
            #             reactants_score = 0.
            #             for reactant in reactants:
            #                 (smi, cur_score) = get_score_from_smi(reactant)
            #                 reactants_smi.append(smi)
            #                 reactants_score = max(reactants_score, cur_score)
            #             products = prompt.strip().split('>>')[1].split('.')
            #             products_smi = []
            #             products_score = 0.
            #             for product in products:
            #                 (smi, cur_score) = get_score_from_smi(product)
            #                 products_smi.append(smi)
            #                 products_score = max(products_score, cur_score)
            #             smi = '{}>>{}'.format('.'.join(reactants_smi), '.'.join(products_smi))
            #             fid.write('%s %s %.4f %.4f %.4f\n' % (prompt.strip(), smi, reactants_score, products_score, products_score-reactants_score))
            #         else: # single or list of mols
            #             reactants = prompt.strip().split('.')
            #             reactants_smi = []
            #             reactants_score = 0.
            #             for reactant in reactants:
            #                 (smi, cur_score) = get_score_from_smi(reactant)
            #                 reactants_smi.append(smi)
            #                 reactants_score = max(reactants_score, cur_score)
            #             fid.write('%s %s %.4f\n' % (prompt.strip(), '.'.join(reactants_smi), reactants_score))

            #     except KeyboardInterrupt:
            #         print('Breaking out of prompt')
            #         fid.close()
            #         raise KeyboardInterrupt
            #     except Exception as e:
            #         print(e)
            #         fid.write('%s\n' % prompt.strip())
            #         continue
        elif test:
            while queue.qsize() == 0:
                print('Letting queue fill up (5 s...)')
                time.sleep(5)

            summarystring = ''
            ctr = 0.0
            if verbose_test: 
                learned_scores = []

            sum_diff_is_pos = 0.0
            sum_diff_is_big = 0.0
            sum_diff = 0.0
            sum_gnorm = 0.0
            sum_loss = 0.0
            while True:
                try:
                    (src_info) = queue.get(timeout=600)
                    if src_info is None:
                        raise Empty
                    cur_topk, cur_score, pnorm, gnorm, cur_loss = session.run([topk, score, param_norm, grad_norm, loss], feed_dict={_lr:lr})
                    
                    it += 1

                    # Padded
                    src_info = [_ for _ in src_info if _]
                    ctr += len(src_info)
                    if len(src_info) < batch_size:
                        print('Found an incomplete batch with only length...')
                        print(len(src_info))

                    # print(src_info[0][4])
                    # print(list(cur_topk[0,:10]))
                    # print('')

                    sum_acc1 += sum([int(src_info[i][4]) in list(cur_topk[i,:1]) for i in range(len(src_info))])
                    sum_acc5 += sum([int(src_info[i][4]) in list(cur_topk[i,:5]) for i in range(len(src_info))])
                    sum_acc10 += sum([int(src_info[i][4]) in list(cur_topk[i,:10]) for i in range(len(src_info))])
                    sum_acc20 += sum([int(src_info[i][4]) in list(cur_topk[i,:20]) for i in range(len(src_info))])
                    sum_acc50 += sum([int(src_info[i][4]) in list(cur_topk[i,:50]) for i in range(len(src_info))])
                    sum_acc100 += sum([int(src_info[i][4]) in list(cur_topk[i,:100]) for i in range(len(src_info))])
                    sum_gnorm += gnorm
                    sum_loss += cur_loss
                        
                    # if verbose_test:
                    #     for i in range(len(ids_batch)):
                    #         learned_scores.append(cur_score[2*i])
                    #         learned_scores.append(cur_score[i*2+1])

                    if it % report_interval == 0:
                        summarystring = "[%09i prods seen], Acc1: %.3f, Acc5: %.3f, Acc10: %.3f, Acc20: %.3f, Acc50: %.3f, Acc100: %.3f, PNorm: %.2f, GNorm: %.2f, Loss: %.4f" % \
                            (ctr, sum_acc1 / float(ctr), 
                            sum_acc5 / float(ctr),
                            sum_acc10 / float(ctr),
                            sum_acc20 / float(ctr),
                            sum_acc50 / float(ctr),
                            sum_acc100 / float(ctr),
                            pnorm, sum_gnorm / float(ctr),
                            sum_loss / float(ctr)) 

                        print(summarystring)
                        sys.stdout.flush()

                except Empty:
                    print('End of data queue I think...have seen {} examples'.format(ctr))
                    break

            summarystring = "[%09i prods seen], Acc1: %.3f, Acc5: %.3f, Acc10: %.3f, Acc20: %.3f, Acc50: %.3f, Acc100: %.3f, PNorm: %.2f, GNorm: %.2f, Loss: %.4f" % \
                (ctr, sum_acc1 / float(ctr), 
                sum_acc5 / float(ctr),
                sum_acc10 / float(ctr),
                sum_acc20 / float(ctr),
                sum_acc50 / float(ctr),
                sum_acc100 / float(ctr),
                pnorm, sum_gnorm / float(ctr),
                sum_loss / float(ctr)) 

            print(summarystring)
            sys.stdout.flush()
            fidsum = open(summary_path, 'a')
            fidsum.write('[%s = %s] %s\n' % (opts.checkpoint, opts.test, summarystring))
            fidsum.close()

            # if verbose_test: 
            #     fid = h5py.File(test_path + '.h5', 'w')
            #     dset = fid.create_dataset('learned_scores', (len(learned_scores),), dtype=np.float32)
            #     dset[:] = np.array(learned_scores)
            #     fid.close()
        else:
            hist_fid = open(opts.save_path + "/model.hist", "a")

            print('Letting queue fill up (10 s)')
            time.sleep(10)
            
            while not coord.should_stop():               
                it += 1
                _, cur_topk, cur_score, pnorm, gnorm, cur_loss = session.run([backprop, topk, score, param_norm, grad_norm, loss], feed_dict={_lr:lr})
                (src_info) = queue.get()
                
                # print(src_info[0][4])
                # print(list(cur_topk[0, :5]))
                sum_acc1 += sum([int(src_info[i][4]) in list(cur_topk[i,:1]) for i in range(len(src_info))])
                sum_acc5 += sum([int(src_info[i][4]) in list(cur_topk[i,:5]) for i in range(len(src_info))])
                sum_acc10 += sum([int(src_info[i][4]) in list(cur_topk[i,:10]) for i in range(len(src_info))])
                sum_acc20 += sum([int(src_info[i][4]) in list(cur_topk[i,:20]) for i in range(len(src_info))])
                sum_acc50 += sum([int(src_info[i][4]) in list(cur_topk[i,:50]) for i in range(len(src_info))])
                sum_acc100 += sum([int(src_info[i][4]) in list(cur_topk[i,:100]) for i in range(len(src_info))])
                sum_gnorm += gnorm
                sum_loss += cur_loss

                # print(sum_acc1)
                # print(batch_size)
                # print(it)
                # print(report_interval)
                # print(sum_acc5)
                # print(sum_acc10)
                # print(sum_acc20)
                # print(sum_acc50)
                # # print(sum_loss)
                # print(cur_loss)

                # print(type(sum_acc1))
                # print(type(batch_size))
                # print(type(it))
                # print(type(report_interval))
                # print(type(sum_acc5))
                # print(type(sum_acc10))
                # print(type(sum_acc20))
                # print(type(sum_acc50))
                # print(type(sum_loss))
                # print(type(pnorm))
                # print(type(sum_gnorm))
                 
                if it % min(report_interval, save_interval) == 0:
                    logstr = "it %06i [%09i prods seen], Acc1: %.3f, Acc5: %.3f, Acc10: %.3f, Acc20: %.3f, Acc50: %.3f, Acc100: %.3f, PNorm: %.2f, GNorm: %.2f, Loss: %.4f" % \
                        (it, it*batch_size, sum_acc1 / float(report_interval * batch_size), 
                            sum_acc5 / float(report_interval * batch_size),
                            sum_acc10 / float(report_interval * batch_size),
                            sum_acc20 / float(report_interval * batch_size),
                            sum_acc50 / float(report_interval * batch_size),
                            sum_acc100 / float(report_interval * batch_size),
                            pnorm, sum_gnorm / report_interval,
                            sum_loss / report_interval) 
                    hist_fid.write(logstr + "\n")
                    print(logstr)
                    sys.stdout.flush()
                    sum_gnorm = 0.0
                    sum_loss = 0.0
                    sum_loss = 0.0;
                    sum_acc1 = 0.0;
                    sum_acc5 = 0.0;
                    sum_acc10 = 0.0;
                    sum_acc20 = 0.0;
                    sum_acc50 = 0.0;
                    sum_acc100 = 0.0;

                    # print('Ex: {:.2f}>>{:.2f} -> diff = {:.2f}'.format(
                    #     cur_score[0], cur_score[1], cur_diff[0]))
                    # print('Ex: ID{} === {}>>{}'.format(
                    #     ids_batch[0], src_batch[0], src_batch[1]))

                    sys.stdout.flush()

                if it % save_interval == 0:
                    lr *= 0.9
                    saver.save(session, opts.save_path + "/model.ckpt", global_step=it)
                    print "Model Saved! Decaying learning rate"

                if it >= max(min_iterations, max_save * save_interval):
                    coord.request_stop()

    except Exception as e:
        print e
        coord.request_stop(e)
    finally:
        if not test and not interactive_mode: 
            saver.save(session, opts.save_path + "/model.final")
            hist_fid.close()
        coord.request_stop()
        coord.join(all_threads)
        try:
            [p.join() for p in processes]
        except Exception:
            pass