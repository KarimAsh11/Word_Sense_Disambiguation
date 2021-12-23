from pre_tf import prepare_mask_infinity, calc_f1
from utility import load_pickle, save_pickle, encode_input_to_id, encode_sense_to_id, encode_instance_mask, encode_elmo
import tensorflow as tf
import numpy as np
from collections import defaultdict
from tf_model import create_tensorflow_model, exp_tf_model
from tf_model_multi import multitask_elmo_tf_model, multitask_seq_to_seq, multitask_bilstm_tagger

#Load pickles
corp_token	           = load_pickle("pickles/corpa/corp_token.pickle")
corp_lemma_pos         = load_pickle("pickles/corpa/corp_lemma_pos.pickle")
corp_fine_grain        = load_pickle("pickles/corpa/corp_fine_grain.pickle")
corp_coarse_domain     = load_pickle("pickles/corpa/corp_coarse_domain.pickle")
corp_coarse_lex        = load_pickle("pickles/corpa/corp_coarse_lex.pickle")
corp_instance      	   = load_pickle("pickles/corpa/corp_instance.pickle")

dev_corp_token	       = load_pickle("pickles/corpa/dev_corp_token.pickle")
dev_corp_lemma_pos     = load_pickle("pickles/corpa/dev_corp_lemma_pos.pickle")
dev_corp_fine_grain    = load_pickle("pickles/corpa/dev_corp_fine_grain.pickle")
dev_corp_coarse_domain = load_pickle("pickles/corpa/dev_corp_coarse_domain.pickle")
dev_corp_coarse_lex    = load_pickle("pickles/corpa/dev_corp_coarse_lex.pickle")
dev_corp_instance      = load_pickle("pickles/corpa/dev_corp_instance.pickle")
 
lemma_pos_id_dict      = load_pickle("pickles/id_dicts/lemma_pos_id_dict.pickle")
fine_senses_id_dict    = load_pickle("pickles/id_dicts/fine_senses_id_dict.pickle")
coarse_domain_id_dict  = load_pickle("pickles/id_dicts/coarse_domain_id_dict.pickle")
coarse_lex_id_dict     = load_pickle("pickles/id_dicts/coarse_lex_id_dict.pickle")

tokens_set     	 	   = load_pickle("pickles/sets/tokens_set.pickle")
fine_inf_id_dict 	   = load_pickle("pickles/cand_dicts/all_id_cand_fine_senses.pickle")
dom_inf_id_dict  	   = load_pickle("pickles/cand_dicts/all_id_cand_coarse_domain.pickle")
lex_inf_id_dict  	   = load_pickle("pickles/cand_dicts/all_id_cand_coarse_lex.pickle")



#CONSTANTS
MAX_LENGTH       = 60
EMBEDDING_SIZE   = 200
HIDDEN_SIZE      = 256
epochs           = 10 
batch_size       = 128
VOCAB_SIZE       = len(lemma_pos_id_dict)
FINE_CLASSES     = len(fine_senses_id_dict)
DOM_CLASSES      = len(coarse_domain_id_dict)
LEX_CLASSES      = len(coarse_lex_id_dict)
print("fine classes:", FINE_CLASSES)
print("Domain classes:", DOM_CLASSES)
print("Lexname classes:", LEX_CLASSES)


elmo_input              = encode_elmo(corp_token, MAX_LENGTH)
index_input             = encode_input_to_id(corp_lemma_pos, lemma_pos_id_dict, MAX_LENGTH)   

index_fine_labels       = encode_sense_to_id(corp_fine_grain, fine_senses_id_dict, tokens_set, MAX_LENGTH)		
index_dom_labels        = encode_sense_to_id(corp_coarse_domain, coarse_domain_id_dict, tokens_set, MAX_LENGTH)		
index_lex_labels        = encode_sense_to_id(corp_coarse_lex, coarse_lex_id_dict, tokens_set, MAX_LENGTH)
index_instance          = encode_instance_mask(corp_instance, MAX_LENGTH)	

dev_elmo_input          = encode_elmo(dev_corp_token, MAX_LENGTH)
dev_index_input         = encode_input_to_id(dev_corp_lemma_pos, lemma_pos_id_dict, MAX_LENGTH)
   
dev_index_fine_labels   = encode_sense_to_id(dev_corp_fine_grain, fine_senses_id_dict, tokens_set, MAX_LENGTH)		
dev_index_dom_labels    = encode_sense_to_id(dev_corp_coarse_domain, coarse_domain_id_dict, tokens_set, MAX_LENGTH)		
dev_index_lex_labels    = encode_sense_to_id(dev_corp_coarse_lex, coarse_lex_id_dict, tokens_set, MAX_LENGTH)		
dev_index_instance      = encode_instance_mask(dev_corp_instance, MAX_LENGTH)		


def batch_generator(X_elmo, X, Y_fine, Y_dom, Y_lex, inst_mask, batch_size, shuffle=False):
	if not shuffle:
		for start in range(0, len(X), batch_size):
			end = start + batch_size
			inf_fine_mask = prepare_mask_infinity(X[start:end], inst_mask[start:end], fine_inf_id_dict, MAX_LENGTH, FINE_CLASSES)
			inf_dom_mask  = prepare_mask_infinity(X[start:end], inst_mask[start:end], dom_inf_id_dict, MAX_LENGTH, DOM_CLASSES)
			inf_lex_mask  = prepare_mask_infinity(X[start:end], inst_mask[start:end], lex_inf_id_dict, MAX_LENGTH, LEX_CLASSES)
			yield X_elmo[start:end], X[start:end], Y_fine[start:end], Y_dom[start:end], Y_lex[start:end], inst_mask[start:end], inf_fine_mask, inf_dom_mask, inf_lex_mask
	else:
		perm = np.random.permutation(len(X))
		for start in range(0, len(X), batch_size):
			end = start + batch_size
			inf_fine_mask = prepare_mask_infinity(X[perm[start:end]], inst_mask[perm[start:end]], fine_inf_id_dict, MAX_LENGTH, FINE_CLASSES)
			inf_dom_mask  = prepare_mask_infinity(X[perm[start:end]], inst_mask[perm[start:end]], dom_inf_id_dict, MAX_LENGTH, DOM_CLASSES)
			inf_lex_mask  = prepare_mask_infinity(X[perm[start:end]], inst_mask[perm[start:end]], lex_inf_id_dict, MAX_LENGTH, LEX_CLASSES)
			yield X_elmo[perm[start:end]], X[perm[start:end]], Y_fine[perm[start:end]], Y_dom[perm[start:end]], Y_lex[perm[start:end]], inst_mask[perm[start:end]], inf_fine_mask, inf_dom_mask, inf_lex_mask



def add_summary(writer, name, value, global_step):
	summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
	writer.add_summary(summary, global_step=global_step)


n_iterations     = int(np.ceil(len(index_input)/batch_size))
n_dev_iterations = int(np.ceil(len(dev_index_input)/batch_size))                        

inputs, fine_labels, dom_labels, lex_labels, fine_inf_mask, dom_inf_mask, lex_inf_mask, instance_mask, keep_prob, loss, train_op, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels = \
	multitask_seq_to_seq(EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, DOM_CLASSES, LEX_CLASSES, MAX_LENGTH)

saver = tf.train.Saver()
with tf.Session() as sess:
	print("\nStarting training...")
	sess.run(tf.global_variables_initializer())
	sess.run(tf.initializers.local_variables())
	train_writer = tf.summary.FileWriter('logging/tensorflow_model', sess.graph)

	for epoch in range(epochs):
		print("\nEpoch", epoch + 1)
		epoch_loss, epoch_acc, epoch_fine_f1, epoch_dom_f1, epoch_lex_f1 = 0., 0., 0., 0., 0.
		mb = 0
		print("======="*10)

		for batch_elmo, batch_x, batch_fine, batch_dom, batch_lex, batch_instance, batch_inf_fine_mask, batch_inf_dom_mask, batch_inf_lex_mask in batch_generator(elmo_input, index_input, index_fine_labels, index_dom_labels, index_lex_labels, index_instance, batch_size, shuffle=True):
			mb += 1

			_, loss_val, acc_val, fine_pred_val, dom_pred_val, lex_pred_val, fine_grd_truth, dom_grd_truth, lex_grd_truth = sess.run([train_op, loss, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels], 
			                                feed_dict={inputs: batch_elmo, fine_labels: batch_fine, dom_labels: batch_dom, lex_labels: batch_lex, fine_inf_mask: batch_inf_fine_mask, dom_inf_mask: batch_inf_dom_mask, lex_inf_mask: batch_inf_lex_mask, instance_mask: batch_instance, keep_prob: 0.90})
			fine_f1_score   = calc_f1(fine_pred_val, fine_grd_truth, fine_senses_id_dict)
			dom_f1_score    = calc_f1(dom_pred_val, dom_grd_truth, coarse_domain_id_dict)
			lex_f1_score    = calc_f1(lex_pred_val, lex_grd_truth, coarse_lex_id_dict)
			
			epoch_fine_f1  += fine_f1_score
			epoch_dom_f1   += dom_f1_score
			epoch_lex_f1   += lex_f1_score
			epoch_loss     += loss_val
			epoch_acc      += acc_val

			print("Epoch:{} - {:.2f}\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f}\tTrain Fine F1: {:.4f}\tTrain Domain F1: {:.4f}\tTrain Lexname F1: {:.4f} ".format(epoch + 1, 100.*mb/n_iterations, epoch_loss/mb, epoch_acc/mb, epoch_fine_f1/mb, epoch_dom_f1/mb, epoch_lex_f1/mb), end="\t")

		epoch_fine_f1     /= n_iterations
		epoch_dom_f1 /= n_iterations
		epoch_lex_f1 /= n_iterations
		epoch_loss   /= n_iterations
		epoch_acc    /= n_iterations
		add_summary(train_writer, "epoch_fine_f1", epoch_fine_f1, epoch)
		add_summary(train_writer, "epoch_dom_f1", epoch_dom_f1, epoch)
		add_summary(train_writer, "epoch_lex_f1", epoch_lex_f1, epoch)
		add_summary(train_writer, "epoch_loss", epoch_loss, epoch)
		add_summary(train_writer, "epoch_acc", epoch_acc, epoch)
		print("\n")
		print("\nEpoch", epoch + 1)
		print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tTrain Fine F1: {:.4f}\tTrain Domain F1: {:.4f}\tTrain Lex F1: {:.4f}".format(epoch_loss, epoch_acc, epoch_fine_f1, epoch_dom_f1, epoch_lex_f1))
		print("======="*10)
		#Save model after each epoch
		model_path = "./tmp/epoch_"+str(epoch+1)+"_model.ckpt"
		print("saving:"+str(epoch+1)+"_model.ckpt")
		save_path  = saver.save(sess, model_path)

		# # DEV - EVALUATION
		dev_loss, dev_acc , dev_fine_f1, dev_dom_f1, dev_lex_f1 = 0.0, 0.0, 0.0, 0.0, 0.0                                                              
		for  batch_elmo, batch_x, batch_fine, batch_dom, batch_lex, batch_instance, batch_inf_fine_mask, batch_inf_dom_mask, batch_inf_lex_mask in batch_generator(dev_elmo_input, dev_index_input, dev_index_fine_labels, dev_index_dom_labels, dev_index_lex_labels, dev_index_instance, batch_size, shuffle=False):
			loss_val, acc_val, fine_pred_val, dom_pred_val, lex_pred_val, fine_grd_truth, dom_grd_truth, lex_grd_truth = sess.run([loss, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels], feed_dict={inputs: batch_elmo, fine_labels: batch_fine, dom_labels: batch_dom, lex_labels: batch_lex, fine_inf_mask: batch_inf_fine_mask, dom_inf_mask: batch_inf_dom_mask, lex_inf_mask: batch_inf_lex_mask, instance_mask: batch_instance, keep_prob: 1})
			fine_f1_val  = calc_f1(fine_pred_val, fine_grd_truth, fine_senses_id_dict)
			dom_f1_score = calc_f1(dom_pred_val, dom_grd_truth, coarse_domain_id_dict)
			lex_f1_score = calc_f1(lex_pred_val, lex_grd_truth, coarse_lex_id_dict)
			
			dev_fine_f1 += fine_f1_val
			dev_dom_f1  += dom_f1_score
			dev_lex_f1  += lex_f1_score
			dev_loss    += loss_val
			dev_acc     += acc_val


		dev_fine_f1  /= n_dev_iterations
		dev_dom_f1   /= n_dev_iterations
		dev_lex_f1   /= n_dev_iterations
		dev_loss     /= n_dev_iterations
		dev_acc      /= n_dev_iterations

		add_summary(train_writer, "epoch_val_fine_f1", dev_fine_f1, epoch)
		add_summary(train_writer, "epoch_val_dom_f1", dev_dom_f1, epoch)
		add_summary(train_writer, "epoch_val_dom_f1", dev_lex_f1, epoch)
		add_summary(train_writer, "epoch_val_loss", dev_loss, epoch)
		add_summary(train_writer, "epoch_val_acc", dev_acc, epoch)
		print("\nDev Loss: {:.4f}\tDev Accuracy: {:.4f}\tDev Fine F1: {:.4f}\tDev Domain F1: {:.4f}\tDev Lex F1: {:.4f}".format(dev_loss, dev_acc, dev_fine_f1, dev_dom_f1, dev_lex_f1))
	train_writer.close()

	save_path = saver.save(sess, "./tmp/final_model.ckpt")




	# DEV - EVALUATION
	dev_loss, dev_acc, dev_fine_f1, dev_dom_f1, dev_lex_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
	for batch_elmo, batch_x, batch_fine, batch_dom, batch_lex, batch_instance, batch_inf_fine_mask, batch_inf_dom_mask, batch_inf_lex_mask in batch_generator(dev_elmo_input, dev_index_input, dev_index_fine_labels, dev_index_dom_labels, dev_index_lex_labels, dev_index_instance, batch_size, shuffle=False):
		loss_val, acc_val, fine_pred_val, dom_pred_val, lex_pred_val, fine_grd_truth, dom_grd_truth, lex_grd_truth = sess.run([loss, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels], feed_dict={inputs: batch_elmo, fine_labels: batch_fine, dom_labels: batch_dom, lex_labels: batch_lex, fine_inf_mask: batch_inf_fine_mask, dom_inf_mask: batch_inf_dom_mask, lex_inf_mask: batch_inf_lex_mask, instance_mask: batch_instance, keep_prob: 1})

		fine_f1_val  = calc_f1(fine_pred_val, fine_grd_truth, fine_senses_id_dict)
		dom_f1_val   = calc_f1(dom_pred_val, dom_grd_truth, coarse_domain_id_dict)
		lex_f1_val   = calc_f1(lex_pred_val, lex_grd_truth, coarse_lex_id_dict)
		

		dev_fine_f1 += fine_f1_val
		dev_dom_f1  += dom_f1_val
		dev_lex_f1  += lex_f1_val			
		dev_loss    += loss_val
		dev_acc     += acc_val
	
	dev_fine_f1  /= n_dev_iterations
	dev_dom_f1   /= n_dev_iterations
	dev_lex_f1   /= n_dev_iterations
	dev_loss     /= n_dev_iterations
	dev_acc      /= n_dev_iterations

	add_summary(train_writer, "epoch_val_fine_f1", dev_fine_f1, epoch)
	add_summary(train_writer, "epoch_val_dom_f1", dev_dom_f1, epoch)
	add_summary(train_writer, "epoch_val_dom_f1", dev_lex_f1, epoch)
	add_summary(train_writer, "epoch_val_loss", dev_loss, epoch)
	add_summary(train_writer, "epoch_val_acc", dev_acc, epoch)
	print("\nDev Loss: {:.4f}\tDev Accuracy: {:.4f}\tDev Fine F1: {:.4f}\tDev Domain F1: {:.4f}\tDev Lex F1: {:.4f}".format(dev_loss, dev_acc, dev_fine_f1, dev_dom_f1, dev_lex_f1))


	#TEST EVALUATION
	print("\nEvaluating test...")
	n_test_iterations = int(np.ceil(len(dev_index_input)/batch_size))
	test_loss, test_acc, test_fine_f1, test_dom_f1, test_lex_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
	for batch_elmo, batch_x, batch_fine, batch_dom, batch_lex, batch_instance, batch_inf_fine_mask, batch_inf_dom_mask, batch_inf_lex_mask in batch_generator(dev_elmo_input, dev_index_input, dev_index_fine_labels, dev_index_dom_labels, dev_index_lex_labels, dev_index_instance, batch_size, shuffle=False):
	
		loss_test, acc_test, fine_pred_val, dom_pred_val, lex_pred_val, fine_grd_truth, dom_grd_truth, lex_grd_truth = sess.run([loss, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels], feed_dict={inputs: batch_elmo, fine_labels: batch_fine, dom_labels: batch_dom, lex_labels: batch_lex, fine_inf_mask: batch_inf_fine_mask, dom_inf_mask: batch_inf_dom_mask, lex_inf_mask: batch_inf_lex_mask, instance_mask: batch_instance, keep_prob: 1})
		

		fine_f1_test  = calc_f1(fine_pred_val, fine_grd_truth, fine_senses_id_dict)
		dom_f1_test   = calc_f1(dom_pred_val, dom_grd_truth, coarse_domain_id_dict)
		lex_f1_test   = calc_f1(lex_pred_val, lex_grd_truth, coarse_lex_id_dict)
		
		test_fine_f1 += fine_f1_test
		test_dom_f1  += dom_f1_test
		test_lex_f1  += lex_f1_test
		test_loss    += loss_test
		test_acc     += acc_test

	test_fine_f1 /= n_test_iterations
	test_dom_f1  /= n_test_iterations
	test_lex_f1  /= n_test_iterations
	test_loss    /= n_test_iterations
	test_acc     /= n_test_iterations
	
	print("\nTest Loss: {:.4f}\tTest Accuracy: {:.4f}\tTest Fine F1: {:.4f}\tTest Domain F1: {:.4f}\tTest Lex F1: {:.4f}".format(test_loss, test_acc, test_fine_f1, test_dom_f1, test_lex_f1))
