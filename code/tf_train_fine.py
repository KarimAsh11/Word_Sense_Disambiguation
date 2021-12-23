from pre_tf import prepare_mask_infinity, infinity_dict, inf_dict_ID, clean_id_dict, calc_f1
from utility import load_pickle, save_pickle, encode_input_to_id, encode_sense_to_id, encode_instance_mask
import tensorflow as tf
import numpy as np
from collections import defaultdict
from tf_model_fine import fine_seq_to_seq_attention_model, fine_attention_model, fine_base_tf_model
#Load inputs and outputs
corp_lemma_pos      = load_pickle("pickles/corpa/corp_lemma_pos.pickle")
corp_fine_grain     = load_pickle("pickles/corpa/corp_fine_grain.pickle")
corp_instance       = load_pickle("pickles/corpa/corp_instance.pickle")

dev_corp_lemma_pos  = load_pickle("pickles/corpa/dev_corp_lemma_pos.pickle")
dev_corp_fine_grain = load_pickle("pickles/corpa/dev_corp_fine_grain.pickle")
dev_corp_instance   = load_pickle("pickles/corpa/dev_corp_instance.pickle")

lemma_pos_id_dict   = load_pickle("pickles/id_dicts/lemma_pos_id_dict.pickle")
fine_senses_id_dict = load_pickle("pickles/id_dicts/fine_senses_id_dict.pickle")

tokens_set     		 = load_pickle("pickles/sets/tokens_set.pickle")
cand_fine_senses     = load_pickle("pickles/cand_dicts/id_cand_fine_senses.pickle")
id_cand_fine_senses  = load_pickle("pickles/cand_dicts/all_id_cand_fine_senses.pickle")

fine_inf_id_dict     = id_cand_fine_senses

#CONSTANTS
MAX_LENGTH       = 60
EMBEDDING_SIZE   = 200
HIDDEN_SIZE      = 128
epochs           = 10 
batch_size       = 128
VOCAB_SIZE       = len(lemma_pos_id_dict)
FINE_CLASSES     = len(fine_senses_id_dict)
print("fine classes:", FINE_CLASSES)

index_input        = encode_input_to_id(corp_lemma_pos, lemma_pos_id_dict, MAX_LENGTH)
   
index_output       = encode_sense_to_id(corp_fine_grain, fine_senses_id_dict, tokens_set, MAX_LENGTH)		

index_instance     = encode_instance_mask(corp_instance, MAX_LENGTH)	

dev_index_input    = encode_input_to_id(dev_corp_lemma_pos, lemma_pos_id_dict, MAX_LENGTH)
   
dev_index_output   = encode_sense_to_id(dev_corp_fine_grain, fine_senses_id_dict, tokens_set, MAX_LENGTH)		

dev_index_instance = encode_instance_mask(dev_corp_instance, MAX_LENGTH)		


def batch_generator(X, Y, inst_mask, inf_dict, batch_size, shuffle=False):
	if not shuffle:
		for start in range(0, len(X), batch_size):
			end = start + batch_size
			inf_fine_mask = prepare_mask_infinity(X[start:end], inst_mask[start:end], fine_inf_id_dict, MAX_LENGTH, FINE_CLASSES)
			yield X[start:end], Y[start:end], inst_mask[start:end], inf_fine_mask
	else:
		perm = np.random.permutation(len(X))
		for start in range(0, len(X), batch_size):
			end = start + batch_size
			inf_fine_mask = prepare_mask_infinity(X[perm[start:end]], inst_mask[perm[start:end]], fine_inf_id_dict, MAX_LENGTH, FINE_CLASSES)
			yield X[perm[start:end]], Y[perm[start:end]], inst_mask[start:end], inf_fine_mask


def add_summary(writer, name, value, global_step):
	summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
	writer.add_summary(summary, global_step=global_step)


n_iterations     = int(np.ceil(len(index_input)/batch_size))
n_dev_iterations = int(np.ceil(len(dev_index_input)/batch_size))                        

inputs, fine_labels, fine_inf_mask, keep_prob, loss, train_op, acc, predictions, masked_fine_labels, pad_mask, unmasked_pred, instance_mask = fine_seq_to_seq_attention_model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, MAX_LENGTH)

saver = tf.train.Saver()
with tf.Session() as sess:
	print("\nStarting training...")
	sess.run(tf.global_variables_initializer())
	sess.run(tf.initializers.local_variables())
	train_writer = tf.summary.FileWriter('logging/tensorflow_model', sess.graph)

	for epoch in range(epochs):
		print("\nEpoch", epoch + 1)
		epoch_loss, epoch_acc, epoch_f1 = 0., 0., 0.
		mb = 0
		print("======="*10)

		for batch_x, batch_y, batch_instance, inf_fine_mask in batch_generator(index_input, index_output, index_instance, fine_inf_id_dict, batch_size, shuffle=False):
			mb += 1

			_, loss_val, acc_val, pred, grd_truth, pad_val, input_val, fine_val, inf_mask_val, unmasked_pred_val = sess.run([train_op, loss, acc, predictions, masked_fine_labels, pad_mask, inputs, fine_labels, fine_inf_mask, unmasked_pred], 
			                                feed_dict={inputs: batch_x, fine_labels: batch_y, fine_inf_mask: inf_fine_mask, instance_mask: batch_instance, keep_prob: 1})

			fine_f1_score  = calc_f1(pred, grd_truth, fine_senses_id_dict)
			
			epoch_f1   += fine_f1_score
			epoch_loss += loss_val
			epoch_acc  += acc_val

			print("Epoch:{} - {:.2f}\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f}\tTrain F1: {:.4f} ".format(epoch + 1, 100.*mb/n_iterations, epoch_loss/mb, epoch_acc/mb, epoch_f1/mb), end="\t")

		epoch_f1 /= n_iterations
		epoch_loss /= n_iterations
		epoch_acc  /= n_iterations
		add_summary(train_writer, "epoch_f1", epoch_f1, epoch)
		add_summary(train_writer, "epoch_loss", epoch_loss, epoch)
		add_summary(train_writer, "epoch_acc", epoch_acc, epoch)
		print("\n")
		print("\nEpoch", epoch + 1)
		print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tTrain F1: {:.4f}".format(epoch_loss, epoch_acc, epoch_f1))
		print("======="*10)
		#Save model after each epoch
		model_path = "./tmp/epoch_"+str(epoch+1)+"_model.ckpt"
		save_path  = saver.save(sess, model_path)

		# # DEV - EVALUATION
		dev_loss, dev_acc , dev_f1 = 0.0, 0.0, 0.0
		for batch_x, batch_y, batch_instance, inf_fine_mask in batch_generator(dev_index_input, dev_index_output, dev_index_instance, fine_inf_id_dict, batch_size, shuffle=False):
			loss_val, acc_val, pred_val, grd_truth_val = sess.run([loss, acc, predictions, masked_fine_labels], feed_dict={inputs: batch_x, fine_labels: batch_y, instance_mask: batch_instance, fine_inf_mask: inf_fine_mask, keep_prob: 1})
			fine_f1_val  = calc_f1(pred_val, grd_truth_val, fine_senses_id_dict)
			
			dev_f1   += fine_f1_val
			dev_loss += loss_val
			dev_acc  += acc_val
		dev_f1   /= n_dev_iterations
		dev_loss /= n_dev_iterations
		dev_acc  /= n_dev_iterations

		add_summary(train_writer, "epoch_val_f1", dev_f1, epoch)
		add_summary(train_writer, "epoch_val_loss", dev_loss, epoch)
		add_summary(train_writer, "epoch_val_acc", dev_acc, epoch)
		print("\nDev Loss: {:.4f}\tDev Accuracy: {:.4f}\tDev F1: {:.4f}".format(dev_loss, dev_acc, dev_f1))
	train_writer.close()

	save_path = saver.save(sess, "./tmp/final_model.ckpt")




	# DEV - EVALUATION
	dev_loss, dev_acc, dev_f1 = 0.0, 0.0, 0.0
	for batch_x, batch_y, batch_instance, inf_fine_mask in batch_generator(dev_index_input, dev_index_output, dev_index_instance, fine_inf_id_dict, batch_size, shuffle=False):
		loss_val, acc_val, pred_val, grd_truth_val = sess.run([loss, acc, predictions, masked_fine_labels], feed_dict={inputs: batch_x, fine_labels: batch_y, instance_mask: batch_instance, fine_inf_mask: inf_fine_mask, keep_prob: 1})
		fine_f1_val  = calc_f1(pred_val, grd_truth_val, fine_senses_id_dict)
		
		dev_f1   += fine_f1_val
		dev_loss += loss_val
		dev_acc  += acc_val
	dev_f1 /= n_dev_iterations
	dev_loss /= n_dev_iterations
	dev_acc /= n_dev_iterations

	add_summary(train_writer, "epoch_val_f1", dev_f1, epoch)
	add_summary(train_writer, "epoch_val_loss", dev_loss, epoch)
	add_summary(train_writer, "epoch_val_acc", dev_acc, epoch)
	print("\nDev Loss: {:.4f}\tDev Accuracy: {:.4f}\tDev F1: {:.4f}".format(dev_loss, dev_acc, dev_f1))






	#TEST EVALUATION
	print("\nEvaluating test...")
	n_test_iterations = int(np.ceil(len(dev_index_input)/batch_size))
	test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0
	for batch_x, batch_y, batch_instance, inf_fine_mask in batch_generator(dev_index_input, dev_index_output, dev_index_instance, fine_inf_id_dict, batch_size, shuffle=False):
		loss_test, acc_test, pred_test, grd_truth_test = sess.run([loss, acc, predictions, masked_fine_labels], feed_dict={inputs: batch_x, fine_labels: batch_y, instance_mask: batch_instance, fine_inf_mask: inf_fine_mask, keep_prob: 1})
		
		fine_f1_test  = calc_f1(pred_test, grd_truth_test, fine_senses_id_dict)
		test_f1   += fine_f1_test
		test_loss += loss_test
		test_acc  += acc_test
	test_f1   /= n_test_iterations
	test_loss /= n_test_iterations
	test_acc  /= n_test_iterations
	print("\nTest Loss: {:.4f}\tTest Accuracy: {:.4f}\tTest F1: {:.4f}".format(test_loss, test_acc, test_f1))
