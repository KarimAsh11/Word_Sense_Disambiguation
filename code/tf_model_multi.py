import tensorflow as tf
import tensorflow_hub as hub






def multitask_seq_to_seq(embedding_size, hidden_size, fine_classes, dom_classes, lex_classes, max_length):
	print("Creating TENSORFLOW model")
	 
	elmo_hub = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
	# Inputs have (batch_size, timesteps) shape.

	# inputs           = tf.placeholder(tf.int32, shape=[None, max_length])
	inputs           = tf.placeholder(tf.string, shape=[None, max_length])
	# inputs = tf.placeholder(tf.int32, shape=[None, ])
	# input_index      = tf.placeholder(tf.int64, shape=[None, max_length])

	fine_labels      = tf.placeholder(tf.int64, shape=[None, max_length])

	dom_labels       = tf.placeholder(tf.int64, shape=[None, max_length])

	lex_labels       = tf.placeholder(tf.int64, shape=[None, max_length])
	
	instance_mask    = tf.placeholder(tf.float32, shape=[None, max_length])
	
	# Keep_prob is a scalar.
	keep_prob        = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length       = tf.count_nonzero(fine_labels, axis=-1, dtype=tf.int32)
	# Create mask using sequence mask
	# pad_mask      = tf.sequence_mask(seq_length)
	pad_mask         = tf.to_float(tf.not_equal(fine_labels, 0))

	# Inf_masks
	fine_inf_mask    = tf.placeholder(dtype=tf.float32, shape=[None, max_length, fine_classes])
	
	dom_inf_mask     = tf.placeholder(dtype=tf.float32, shape=[None, max_length, dom_classes])
	
	lex_inf_mask     = tf.placeholder(dtype=tf.float32, shape=[None, max_length, lex_classes])

	with tf.variable_scope("embeddings"):
		# embedding_matrix = tf.get_variable("embeddings", shape=[vocab_size, embedding_size])
		# embeddings       = tf.nn.embedding_lookup(embedding_matrix, inputs)
		embeddings       = elmo_hub(inputs={"tokens": inputs, "sequence_len": seq_length}, signature="tokens", as_dict=True)["elmo"]

	with tf.variable_scope("rnn_encoder"):

		rnn_cell_fwd = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd, rnn_cell_bwd, embeddings, sequence_length=seq_length, dtype=tf.float32)
		concat_out = tf.concat(outputs, 2)


	with tf.variable_scope("attention"):

		hidden_size_att = concat_out.shape[2].value
		epsilon = 1e-8


		# Trainable parameters
		omega_att = tf.get_variable(shape=[hidden_size_att, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega')

		omega_H = tf.tensordot(concat_out, omega_att, axes=1) 
		omega_H = tf.squeeze(omega_H, -1) 

		u  = tf.tanh(omega_H)  

		# Softmax
		a  = tf.exp(u)
		a  = a * pad_mask
		a /= tf.reduce_sum(a, axis=1, keepdims=True) + epsilon 

		w_inputs = concat_out * tf.expand_dims(a, -1)

		c        = tf.reduce_sum(w_inputs, axis=1)

		c_expanded      = tf.expand_dims(c,axis=1)
		attention_vec   = tf.tile(c_expanded, [1,max_length,1])
		self_att_concat = tf.concat([concat_out,attention_vec], 2)
		

	with tf.variable_scope("rnn_decoder"):

		rnn_cell_fwd_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd_dec = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd_dec,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd_dec = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd_dec,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs_dec, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd_dec, rnn_cell_bwd_dec, self_att_concat, sequence_length=seq_length, dtype=tf.float32)
		concat_out_dec = tf.concat(outputs_dec, 2)
	

	with tf.variable_scope("attention_domain"):

		hidden_size_att_dom = concat_out_dec.shape[2].value
		epsilon = 1e-8


		# Trainable parameters
		omega_att_dom = tf.get_variable(shape=[hidden_size_att_dom, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega_dom')

		omega_H_dom = tf.tensordot(concat_out_dec, omega_att_dom, axes=1) 
		omega_H_dom = tf.squeeze(omega_H_dom, -1) 

		u_dom  = tf.tanh(omega_H_dom)  

		# Softmax
		a_dom  = tf.exp(u_dom)
		a_dom  = a_dom * pad_mask
		a_dom /= tf.reduce_sum(a_dom, axis=1, keepdims=True) + epsilon 

		w_inputs_dom = concat_out_dec * tf.expand_dims(a_dom, -1)

		c_dom        = tf.reduce_sum(w_inputs_dom, axis=1)

		c_expanded_dom     		= tf.expand_dims(c_dom,axis=1)
		attention_vec_dom   	= tf.tile(c_expanded_dom, [1,max_length,1])
		self_att_concat_dom = tf.concat([concat_out_dec,attention_vec_dom], 2)
		
	
	with tf.variable_scope("domain"):
		dom_logits      = tf.layers.dense(self_att_concat_dom, dom_classes, activation=None)
		inf_dom_logits  = tf.math.add(dom_logits, dom_inf_mask)
		dom_padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dom_labels, logits=inf_dom_logits)
		dom_padded_loss = tf.multiply(dom_padded_loss, instance_mask)
		dom_masked_loss = tf.boolean_mask(dom_padded_loss, pad_mask)
		dom_loss        = tf.reduce_mean(dom_masked_loss)


	with tf.variable_scope("attention_domain"):

		hidden_size_att_lex = concat_out_dec.shape[2].value
		epsilon = 1e-8


		# Trainable parameters
		omega_att_lex = tf.get_variable(shape=[hidden_size_att_lex, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega_lex')

		omega_H_lex = tf.tensordot(concat_out_dec, omega_att_dom, axes=1) 
		omega_H_lex = tf.squeeze(omega_H_lex, -1) 

		u_lex  = tf.tanh(omega_H_lex)  

		# Softmax
		a_lex  = tf.exp(u_lex)
		a_lex  = a_lex * pad_mask
		a_lex /= tf.reduce_sum(a_lex, axis=1, keepdims=True) + epsilon 

		w_inputs_lex = concat_out_dec * tf.expand_dims(a_lex, -1)

		c_lex        = tf.reduce_sum(w_inputs_lex, axis=1)

		c_expanded_lex     		= tf.expand_dims(c_lex,axis=1)
		attention_vec_lex   	= tf.tile(c_expanded_lex, [1,max_length,1])
		self_att_concat_lex = tf.concat([concat_out_dec,attention_vec_lex], 2)
		

	with tf.variable_scope("lexname"):
		lex_logits      = tf.layers.dense(self_att_concat_lex, lex_classes, activation=None)
		inf_lex_logits  = tf.math.add(lex_logits, lex_inf_mask)
		lex_padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lex_labels, logits=inf_lex_logits)
		lex_padded_loss = tf.multiply(lex_padded_loss, instance_mask)
		lex_masked_loss = tf.boolean_mask(lex_padded_loss, pad_mask)
		lex_loss        = tf.reduce_mean(lex_masked_loss)

	with tf.variable_scope("rnn_fine"):

		rnn_cell_fwd_fine = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd_fine = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd_fine = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd_fine,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd_fine = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd_fine,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs_fine, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd_fine, rnn_cell_bwd_fine, concat_out_dec, sequence_length=seq_length, dtype=tf.float32)
		concat_out_fine = tf.concat(outputs_fine, 2)



	with tf.variable_scope("attention_fine"):

		hidden_size_att_fine = concat_out_fine.shape[2].value
		epsilon = 1e-8


		# Trainable parameters
		omega_att_fine = tf.get_variable(shape=[hidden_size_att_fine, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega_fine')

		omega_H_fine = tf.tensordot(concat_out_fine, omega_att_fine, axes=1) 
		omega_H_fine = tf.squeeze(omega_H_fine, -1) 

		u_fine  = tf.tanh(omega_H_fine)  

		# Softmax
		a_fine  = tf.exp(u_fine)
		a_fine  = a_fine * pad_mask
		a_fine /= tf.reduce_sum(a_fine, axis=1, keepdims=True) + epsilon 

		w_inputs_fine = concat_out_fine * tf.expand_dims(a_fine, -1)

		c_fine       = tf.reduce_sum(w_inputs_fine, axis=1)

		c_expanded_fine      = tf.expand_dims(c_fine,axis=1)
		attention_vec_fine   = tf.tile(c_expanded_fine, [1,max_length,1])
		self_att_concat_fine = tf.concat([concat_out_fine,attention_vec_fine], 2)
		
	with tf.variable_scope("dense"):
		logits = tf.layers.dense(self_att_concat_fine, fine_classes, activation=None)

	with tf.variable_scope("infinite_mask"):
		inf_fine_logits = tf.math.add(logits, fine_inf_mask)
		# inf_fine_logits = logits

	with tf.variable_scope("fine_loss"):
		padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fine_labels, logits=inf_fine_logits)
		padded_loss = tf.multiply(padded_loss, instance_mask)
		masked_loss = tf.boolean_mask(padded_loss, pad_mask)
		fine_loss   = tf.reduce_mean(masked_loss)


	with tf.variable_scope("train"):
		# loss = fine_loss + dom_loss + lex_loss
		coarse_loss = tf.add(dom_loss, lex_loss)
		
		loss        = tf.add(fine_loss, coarse_loss)
		
		train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

	with tf.variable_scope("accuracy_fine"):
		unmasked_fine_pred = tf.argmax(inf_fine_logits, axis=-1)
		fine_predictions   = tf.boolean_mask(unmasked_fine_pred, pad_mask)
		masked_fine_labels = tf.boolean_mask(fine_labels, pad_mask)
		masked_fine_acc    = tf.equal(fine_predictions, masked_fine_labels)
		
		fine_acc_mask      = tf.to_float(tf.not_equal(masked_fine_labels, 2))

		fine_acc_pred      = tf.boolean_mask(fine_predictions, fine_acc_mask)
		fine_acc_labels    = tf.boolean_mask(masked_fine_labels, fine_acc_mask)
		fine_test_acc      = tf.equal(fine_acc_pred, fine_acc_labels)

		acc                = tf.reduce_mean(tf.cast(fine_test_acc, tf.float32))

	with tf.variable_scope("accuracy_domain"):
		unmasked_dom_pred  = tf.argmax(inf_dom_logits, axis=-1)
		dom_predictions    = tf.boolean_mask(unmasked_dom_pred, pad_mask)
		masked_dom_labels  = tf.boolean_mask(dom_labels, pad_mask)
		masked_dom_acc     = tf.equal(dom_predictions, masked_fine_labels)
		
		dom_acc_mask       = tf.to_float(tf.not_equal(masked_dom_labels, 2))

		dom_acc_pred       = tf.boolean_mask(dom_predictions, dom_acc_mask)
		dom_acc_labels     = tf.boolean_mask(masked_dom_labels, dom_acc_mask)
		dom_test_acc       = tf.equal(dom_acc_pred, dom_acc_labels)

		dom_acc            = tf.reduce_mean(tf.cast(dom_test_acc, tf.float32))
	
	with tf.variable_scope("accuracy_lexname"):
		unmasked_lex_pred  = tf.argmax(inf_lex_logits, axis=-1)
		lex_predictions    = tf.boolean_mask(unmasked_lex_pred, pad_mask)
		masked_lex_labels  = tf.boolean_mask(lex_labels, pad_mask)
		masked_lex_acc     = tf.equal(lex_predictions, masked_lex_labels)
		
		lex_acc_mask       = tf.to_float(tf.not_equal(masked_lex_labels, 2))

		lex_acc_pred       = tf.boolean_mask(lex_predictions, lex_acc_mask)
		lex_acc_labels     = tf.boolean_mask(masked_lex_labels, lex_acc_mask)
		lex_test_acc       = tf.equal(lex_acc_pred, lex_acc_labels)

		lex_acc            = tf.reduce_mean(tf.cast(lex_test_acc, tf.float32))
	
	return inputs, fine_labels, dom_labels, lex_labels, fine_inf_mask, dom_inf_mask, lex_inf_mask, instance_mask, keep_prob, loss, train_op, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels

















def multitask_bilstm_tagger(embedding_size, hidden_size, fine_classes, dom_classes, lex_classes, max_length):
	print("Creating TENSORFLOW model")
	 
	elmo_hub = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
	# Inputs have (batch_size, timesteps) shape.

	# inputs           = tf.placeholder(tf.int32, shape=[None, max_length])
	inputs           = tf.placeholder(tf.string, shape=[None, max_length])
	# inputs = tf.placeholder(tf.int32, shape=[None, ])
	# input_index      = tf.placeholder(tf.int64, shape=[None, max_length])

	fine_labels      = tf.placeholder(tf.int64, shape=[None, max_length])

	dom_labels       = tf.placeholder(tf.int64, shape=[None, max_length])

	lex_labels       = tf.placeholder(tf.int64, shape=[None, max_length])
	
	instance_mask    = tf.placeholder(tf.float32, shape=[None, max_length])
	
	# Keep_prob is a scalar.
	keep_prob        = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length       = tf.count_nonzero(fine_labels, axis=-1, dtype=tf.int32)
	# Create mask using sequence mask
	# pad_mask      = tf.sequence_mask(seq_length)
	pad_mask         = tf.to_float(tf.not_equal(fine_labels, 0))

	# Inf_masks
	fine_inf_mask    = tf.placeholder(dtype=tf.float32, shape=[None, max_length, fine_classes])
	
	dom_inf_mask     = tf.placeholder(dtype=tf.float32, shape=[None, max_length, dom_classes])
	
	lex_inf_mask     = tf.placeholder(dtype=tf.float32, shape=[None, max_length, lex_classes])

	with tf.variable_scope("embeddings"):
		# embedding_matrix = tf.get_variable("embeddings", shape=[vocab_size, embedding_size])
		# embeddings       = tf.nn.embedding_lookup(embedding_matrix, inputs)
		embeddings       = elmo_hub(inputs={"tokens": inputs, "sequence_len": seq_length}, signature="tokens", as_dict=True)["elmo"]


	with tf.variable_scope("bilstm"):

		rnn_cell_fwd_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd_dec = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd_dec,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd_dec = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd_dec,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs_dec, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd_dec, rnn_cell_bwd_dec, embeddings, sequence_length=seq_length, dtype=tf.float32)
		concat_out_dec = tf.concat(outputs_dec, 2)
	

	with tf.variable_scope("attention_domain"):

		hidden_size_att_dom = concat_out_dec.shape[2].value
		epsilon = 1e-8


		# Trainable parameters
		omega_att_dom = tf.get_variable(shape=[hidden_size_att_dom, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega_dom')

		omega_H_dom = tf.tensordot(concat_out_dec, omega_att_dom, axes=1) 
		omega_H_dom = tf.squeeze(omega_H_dom, -1) 

		u_dom  = tf.tanh(omega_H_dom)  

		# Softmax
		a_dom  = tf.exp(u_dom)
		a_dom  = a_dom * pad_mask
		a_dom /= tf.reduce_sum(a_dom, axis=1, keepdims=True) + epsilon 

		w_inputs_dom = concat_out_dec * tf.expand_dims(a_dom, -1)

		c_dom        = tf.reduce_sum(w_inputs_dom, axis=1)

		c_expanded_dom     		= tf.expand_dims(c_dom,axis=1)
		attention_vec_dom   	= tf.tile(c_expanded_dom, [1,max_length,1])
		self_att_concat_dom = tf.concat([concat_out_dec,attention_vec_dom], 2)
		
	
	with tf.variable_scope("domain"):
		dom_logits      = tf.layers.dense(self_att_concat_dom, dom_classes, activation=None)
		inf_dom_logits  = tf.math.add(dom_logits, dom_inf_mask)
		dom_padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dom_labels, logits=inf_dom_logits)
		dom_padded_loss = tf.multiply(dom_padded_loss, instance_mask)
		dom_masked_loss = tf.boolean_mask(dom_padded_loss, pad_mask)
		dom_loss        = tf.reduce_mean(dom_masked_loss)


	with tf.variable_scope("attention_domain"):

		hidden_size_att_lex = concat_out_dec.shape[2].value
		epsilon = 1e-8


		# Trainable parameters
		omega_att_lex = tf.get_variable(shape=[hidden_size_att_lex, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega_lex')

		omega_H_lex = tf.tensordot(concat_out_dec, omega_att_dom, axes=1) 
		omega_H_lex = tf.squeeze(omega_H_lex, -1) 

		u_lex  = tf.tanh(omega_H_lex)  

		# Softmax
		a_lex  = tf.exp(u_lex)
		a_lex  = a_lex * pad_mask
		a_lex /= tf.reduce_sum(a_lex, axis=1, keepdims=True) + epsilon 

		w_inputs_lex = concat_out_dec * tf.expand_dims(a_lex, -1)

		c_lex        = tf.reduce_sum(w_inputs_lex, axis=1)

		c_expanded_lex     		= tf.expand_dims(c_lex,axis=1)
		attention_vec_lex   	= tf.tile(c_expanded_lex, [1,max_length,1])
		self_att_concat_lex = tf.concat([concat_out_dec,attention_vec_lex], 2)
		

	with tf.variable_scope("lexname"):
		lex_logits      = tf.layers.dense(self_att_concat_lex, lex_classes, activation=None)
		inf_lex_logits  = tf.math.add(lex_logits, lex_inf_mask)
		lex_padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lex_labels, logits=inf_lex_logits)
		lex_padded_loss = tf.multiply(lex_padded_loss, instance_mask)
		lex_masked_loss = tf.boolean_mask(lex_padded_loss, pad_mask)
		lex_loss        = tf.reduce_mean(lex_masked_loss)

		# Sometimes Yes, Somtimes No
		# logits = tf.squeeze(logits)

	with tf.variable_scope("rnn_fine"):

		rnn_cell_fwd_fine = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd_fine = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd_fine = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd_fine,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd_fine = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd_fine,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs_fine, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd_fine, rnn_cell_bwd_fine, concat_out_dec, sequence_length=seq_length, dtype=tf.float32)
		concat_out_fine = tf.concat(outputs_fine, 2)



	with tf.variable_scope("attention_fine"):

		hidden_size_att_fine = concat_out_fine.shape[2].value
		epsilon = 1e-8


		# Trainable parameters
		omega_att_fine = tf.get_variable(shape=[hidden_size_att_fine, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega_fine')

		omega_H_fine = tf.tensordot(concat_out_fine, omega_att_fine, axes=1) 
		omega_H_fine = tf.squeeze(omega_H_fine, -1) 

		u_fine  = tf.tanh(omega_H_fine)  

		# Softmax
		a_fine  = tf.exp(u_fine)
		a_fine  = a_fine * pad_mask
		a_fine /= tf.reduce_sum(a_fine, axis=1, keepdims=True) + epsilon 

		w_inputs_fine = concat_out_fine * tf.expand_dims(a_fine, -1)

		c_fine       = tf.reduce_sum(w_inputs_fine, axis=1)

		c_expanded_fine      = tf.expand_dims(c_fine,axis=1)
		attention_vec_fine   = tf.tile(c_expanded_fine, [1,max_length,1])
		self_att_concat_fine = tf.concat([concat_out_fine,attention_vec_fine], 2)
		
	with tf.variable_scope("dense"):
		logits = tf.layers.dense(self_att_concat_fine, fine_classes, activation=None)

	with tf.variable_scope("infinite_mask"):
		inf_fine_logits = tf.math.add(logits, fine_inf_mask)
		# inf_fine_logits = logits

	with tf.variable_scope("fine_loss"):
		padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fine_labels, logits=inf_fine_logits)
		padded_loss = tf.multiply(padded_loss, instance_mask)
		masked_loss = tf.boolean_mask(padded_loss, pad_mask)
		fine_loss   = tf.reduce_mean(masked_loss)


	with tf.variable_scope("train"):
		# loss = fine_loss + dom_loss + lex_loss
		coarse_loss = tf.add(dom_loss, lex_loss)
		
		loss        = tf.add(fine_loss, coarse_loss)
		
		train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

	with tf.variable_scope("accuracy_fine"):
		unmasked_fine_pred = tf.argmax(inf_fine_logits, axis=-1)
		fine_predictions   = tf.boolean_mask(unmasked_fine_pred, pad_mask)
		masked_fine_labels = tf.boolean_mask(fine_labels, pad_mask)
		masked_fine_acc    = tf.equal(fine_predictions, masked_fine_labels)
		
		fine_acc_mask      = tf.to_float(tf.not_equal(masked_fine_labels, 2))

		fine_acc_pred      = tf.boolean_mask(fine_predictions, fine_acc_mask)
		fine_acc_labels    = tf.boolean_mask(masked_fine_labels, fine_acc_mask)
		fine_test_acc      = tf.equal(fine_acc_pred, fine_acc_labels)

		acc                = tf.reduce_mean(tf.cast(fine_test_acc, tf.float32))

	with tf.variable_scope("accuracy_domain"):
		unmasked_dom_pred  = tf.argmax(inf_dom_logits, axis=-1)
		dom_predictions    = tf.boolean_mask(unmasked_dom_pred, pad_mask)
		masked_dom_labels  = tf.boolean_mask(dom_labels, pad_mask)
		masked_dom_acc     = tf.equal(dom_predictions, masked_fine_labels)
		
		dom_acc_mask       = tf.to_float(tf.not_equal(masked_dom_labels, 2))

		dom_acc_pred       = tf.boolean_mask(dom_predictions, dom_acc_mask)
		dom_acc_labels     = tf.boolean_mask(masked_dom_labels, dom_acc_mask)
		dom_test_acc       = tf.equal(dom_acc_pred, dom_acc_labels)

		dom_acc            = tf.reduce_mean(tf.cast(dom_test_acc, tf.float32))
	
	with tf.variable_scope("accuracy_lexname"):
		unmasked_lex_pred  = tf.argmax(inf_lex_logits, axis=-1)
		lex_predictions    = tf.boolean_mask(unmasked_lex_pred, pad_mask)
		masked_lex_labels  = tf.boolean_mask(lex_labels, pad_mask)
		masked_lex_acc     = tf.equal(lex_predictions, masked_lex_labels)
		
		lex_acc_mask       = tf.to_float(tf.not_equal(masked_lex_labels, 2))

		lex_acc_pred       = tf.boolean_mask(lex_predictions, lex_acc_mask)
		lex_acc_labels     = tf.boolean_mask(masked_lex_labels, lex_acc_mask)
		lex_test_acc       = tf.equal(lex_acc_pred, lex_acc_labels)

		lex_acc            = tf.reduce_mean(tf.cast(lex_test_acc, tf.float32))
	
	return inputs, fine_labels, dom_labels, lex_labels, fine_inf_mask, dom_inf_mask, lex_inf_mask, instance_mask, keep_prob, loss, train_op, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels




























def seq_to_seq(vocab_size, embedding_size, hidden_size, fine_classes, max_length):
	print("Creating TENSORFLOW model")
	 
	# Inputs have (batch_size, timesteps) shape.
	inputs           = tf.placeholder(tf.int32, shape=[None, max_length])

	fine_labels      = tf.placeholder(tf.int64, shape=[None, max_length])

	domain_labels    = tf.placeholder(tf.int64, shape=[None, max_length])
	
	lexname_labels   = tf.placeholder(tf.int64, shape=[None, max_length])

	instance_mask    = tf.placeholder(tf.float32, shape=[None, max_length])
	
	# Keep_prob is a scalar.
	keep_prob        = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length       = tf.count_nonzero(fine_labels, axis=-1)
	# Create mask using sequence mask
	# pad_mask      = tf.sequence_mask(seq_length)
	pad_mask         = tf.to_float(tf.not_equal(fine_labels, 0))

	# Inf_mask
	fine_inf_mask = tf.placeholder(dtype=tf.float32, shape=[None, max_length, fine_classes])

	with tf.variable_scope("uni_embeddings"):
		embedding_matrix = tf.get_variable("uni_embeddings", shape=[vocab_size, embedding_size])
		embeddings       = tf.nn.embedding_lookup(embedding_matrix, inputs)

	with tf.variable_scope("rnn - coarse"):

		rnn_cell_fwd = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd, rnn_cell_bwd, embeddings, sequence_length=seq_length, dtype=tf.float32)
		concat_out = tf.concat(outputs, 2)


	with tf.variable_scope("attention"):

		hidden_size_att = concat_out.shape[2].value
		epsilon = 1e-8


		# Trainable parameters
		omega_att = tf.get_variable(shape=[hidden_size_att, 1], dtype=tf.float32,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05), name='omega')

		omega_H = tf.tensordot(concat_out, omega_att, axes=1) 
		omega_H = tf.squeeze(omega_H, -1) 

		u  = tf.tanh(omega_H)  

		# Softmax
		a  = tf.exp(u)
		a  = a * pad_mask
		a /= tf.reduce_sum(a, axis=1, keepdims=True) + epsilon 

		w_inputs = concat_out * tf.expand_dims(a, -1)

		c        = tf.reduce_sum(w_inputs, axis=1)

		c_expanded      = tf.expand_dims(c,axis=1)
		attention_vec   = tf.tile(c_expanded, [1,max_length,1])
		self_att_concat = tf.concat([concat_out,attention_vec], 2)
		

	with tf.variable_scope("rnn - decoder"):

		rnn_cell_fwd_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd_dec = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd_dec,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd_dec = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd_dec,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs_dec, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd_dec, rnn_cell_bwd_dec, self_att_concat, sequence_length=seq_length, dtype=tf.float32)
		concat_out_dec = tf.concat(outputs_dec, 2)
	
	with tf.variable_scope("dense"):
		logits = tf.layers.dense(concat_out_dec, fine_classes, activation=None)

	with tf.variable_scope("infinite_mask"):
		inf_fine_logits = tf.math.add(logits, fine_inf_mask)

	with tf.variable_scope("loss"):
		padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fine_labels, logits=inf_fine_logits)
		padded_loss = tf.multiply(padded_loss, instance_mask)
		masked_loss = tf.boolean_mask(padded_loss, pad_mask)
		loss        = tf.reduce_mean(masked_loss)

	with tf.variable_scope("train"):
		# train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
		train_op    = tf.train.MomentumOptimizer(0.04, 0.95).minimize(loss)
	with tf.variable_scope("accuracy"):
		unmasked_pred      = tf.argmax(inf_fine_logits, axis=-1)
		predictions        = tf.boolean_mask(unmasked_pred, pad_mask)
		masked_fine_labels = tf.boolean_mask(fine_labels, pad_mask)
		masked_acc         = tf.equal(predictions, masked_fine_labels)
		
		acc_mask           = tf.to_float(tf.not_equal(masked_fine_labels, 2))

		acc_pred           = tf.boolean_mask(predictions, acc_mask)
		acc_labels         = tf.boolean_mask(masked_fine_labels, acc_mask)
		test_acc           = tf.equal(acc_pred, acc_labels)

		acc                = tf.reduce_mean(tf.cast(test_acc, tf.float32))

	return inputs, fine_labels, fine_inf_mask, keep_prob, loss, train_op, acc, predictions, masked_fine_labels, pad_mask, unmasked_pred, instance_mask














#----------------------------------- Multitask base model --------------------------------------
def multitask_elmo_tf_model(vocab_size, embedding_size, hidden_size, fine_classes, dom_classes, lex_classes, max_length):
	print("Creating TENSORFLOW model")
	
	elmo_hub = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
	# Inputs have (batch_size, timesteps) shape.

	# inputs           = tf.placeholder(tf.int32, shape=[None, max_length])
	inputs           = tf.placeholder(tf.string, shape=[None, max_length])
	# inputs = tf.placeholder(tf.int32, shape=[None, ])
	fine_labels      = tf.placeholder(tf.int64, shape=[None, max_length])

	dom_labels       = tf.placeholder(tf.int64, shape=[None, max_length])

	lex_labels       = tf.placeholder(tf.int64, shape=[None, max_length])
	
	instance_mask    = tf.placeholder(tf.float32, shape=[None, max_length])
	
	# Keep_prob is a scalar.
	keep_prob        = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length       = tf.count_nonzero(fine_labels, axis=-1, dtype=tf.int32)
	# Create mask using sequence mask
	# pad_mask      = tf.sequence_mask(seq_length)
	pad_mask         = tf.to_float(tf.not_equal(fine_labels, 0))

	# Inf_mask
	fine_inf_mask    = tf.placeholder(dtype=tf.float32, shape=[None, max_length, fine_classes])
	
	dom_inf_mask     = tf.placeholder(dtype=tf.float32, shape=[None, max_length, dom_classes])
	
	lex_inf_mask     = tf.placeholder(dtype=tf.float32, shape=[None, max_length, lex_classes])

	with tf.variable_scope("embeddings"):
		# embedding_matrix = tf.get_variable("embeddings", shape=[vocab_size, embedding_size])
		# embeddings       = tf.nn.embedding_lookup(embedding_matrix, inputs)
		embeddings       = elmo_hub(inputs={"tokens": inputs, "sequence_len": seq_length}, signature="tokens", as_dict=True)["elmo"]

	with tf.variable_scope("rnn_coarse"):

		rnn_cell_fwd = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd, rnn_cell_bwd, embeddings, sequence_length=seq_length, dtype=tf.float32)
		concat_out = tf.concat(outputs, 2)


	with tf.variable_scope("domain"):
		dom_logits      = tf.layers.dense(concat_out, dom_classes, activation=None)
		inf_dom_logits  = tf.math.add(dom_logits, dom_inf_mask)
		dom_padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dom_labels, logits=inf_dom_logits)
		dom_padded_loss = tf.multiply(dom_padded_loss, instance_mask)
		dom_masked_loss = tf.boolean_mask(dom_padded_loss, pad_mask)
		dom_loss        = tf.reduce_mean(dom_masked_loss)

	with tf.variable_scope("lexname"):
		lex_logits      = tf.layers.dense(concat_out, lex_classes, activation=None)
		inf_lex_logits  = tf.math.add(lex_logits, lex_inf_mask)
		lex_padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lex_labels, logits=inf_lex_logits)
		lex_padded_loss = tf.multiply(lex_padded_loss, instance_mask)
		lex_masked_loss = tf.boolean_mask(lex_padded_loss, pad_mask)
		lex_loss        = tf.reduce_mean(lex_masked_loss)

		# Sometimes Yes, Somtimes No
		# logits = tf.squeeze(logits)

	with tf.variable_scope("rnn_fine"):

		rnn_cell_fwd_fine = tf.nn.rnn_cell.LSTMCell(hidden_size)
		rnn_cell_bwd_fine = tf.nn.rnn_cell.LSTMCell(hidden_size)

		# Add dropout to the LSTM cell
		rnn_cell_fwd_fine = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fwd_fine,
			input_keep_prob=keep_prob, 
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)

		rnn_cell_bwd_fine = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bwd_fine,
			input_keep_prob=keep_prob,
			output_keep_prob=keep_prob,
			state_keep_prob=keep_prob)
		outputs_fine, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd_fine, rnn_cell_bwd_fine, concat_out, sequence_length=seq_length, dtype=tf.float32)
		concat_out_fine = tf.concat(outputs_fine, 2)

	with tf.variable_scope("dense"):
		logits = tf.layers.dense(concat_out_fine, fine_classes, activation=None)

	with tf.variable_scope("infinite_mask"):
		inf_fine_logits = tf.math.add(logits, fine_inf_mask)
		# inf_fine_logits = logits

	with tf.variable_scope("fine_loss"):
		padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fine_labels, logits=inf_fine_logits)
		padded_loss = tf.multiply(padded_loss, instance_mask)
		masked_loss = tf.boolean_mask(padded_loss, pad_mask)
		fine_loss   = tf.reduce_mean(masked_loss)


	with tf.variable_scope("train"):
		# loss = fine_loss + dom_loss + lex_loss
		coarse_loss = tf.add(dom_loss, lex_loss)
		
		loss        = tf.add(fine_loss, coarse_loss)
		
		train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

	with tf.variable_scope("accuracy_fine"):
		unmasked_fine_pred = tf.argmax(inf_fine_logits, axis=-1)
		fine_predictions   = tf.boolean_mask(unmasked_fine_pred, pad_mask)
		masked_fine_labels = tf.boolean_mask(fine_labels, pad_mask)
		masked_fine_acc    = tf.equal(fine_predictions, masked_fine_labels)
		
		fine_acc_mask      = tf.to_float(tf.not_equal(masked_fine_labels, 2))

		fine_acc_pred      = tf.boolean_mask(fine_predictions, fine_acc_mask)
		fine_acc_labels    = tf.boolean_mask(masked_fine_labels, fine_acc_mask)
		fine_test_acc      = tf.equal(fine_acc_pred, fine_acc_labels)

		acc                = tf.reduce_mean(tf.cast(fine_test_acc, tf.float32))

	with tf.variable_scope("accuracy_domain"):
		unmasked_dom_pred  = tf.argmax(inf_dom_logits, axis=-1)
		dom_predictions    = tf.boolean_mask(unmasked_dom_pred, pad_mask)
		masked_dom_labels  = tf.boolean_mask(dom_labels, pad_mask)
		masked_dom_acc     = tf.equal(dom_predictions, masked_fine_labels)
		
		dom_acc_mask       = tf.to_float(tf.not_equal(masked_dom_labels, 2))

		dom_acc_pred       = tf.boolean_mask(dom_predictions, dom_acc_mask)
		dom_acc_labels     = tf.boolean_mask(masked_dom_labels, dom_acc_mask)
		dom_test_acc       = tf.equal(dom_acc_pred, dom_acc_labels)

		dom_acc            = tf.reduce_mean(tf.cast(dom_test_acc, tf.float32))
	
	with tf.variable_scope("accuracy_lexname"):
		unmasked_lex_pred  = tf.argmax(inf_lex_logits, axis=-1)
		lex_predictions    = tf.boolean_mask(unmasked_lex_pred, pad_mask)
		masked_lex_labels  = tf.boolean_mask(lex_labels, pad_mask)
		masked_lex_acc     = tf.equal(lex_predictions, masked_lex_labels)
		
		lex_acc_mask       = tf.to_float(tf.not_equal(masked_lex_labels, 2))

		lex_acc_pred       = tf.boolean_mask(lex_predictions, lex_acc_mask)
		lex_acc_labels     = tf.boolean_mask(masked_lex_labels, lex_acc_mask)
		lex_test_acc       = tf.equal(lex_acc_pred, lex_acc_labels)

		lex_acc            = tf.reduce_mean(tf.cast(lex_test_acc, tf.float32))
	
	return inputs, fine_labels, dom_labels, lex_labels, fine_inf_mask, dom_inf_mask, lex_inf_mask, instance_mask, keep_prob, loss, train_op, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels




