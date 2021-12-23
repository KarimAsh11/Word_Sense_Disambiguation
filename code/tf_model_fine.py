import tensorflow as tf
import tensorflow_hub as hub





def fine_seq_to_seq_attention_model(vocab_size, embedding_size, hidden_size, fine_classes, max_length):
	print("Creating TENSORFLOW model")
	# elmo_hub = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
	# inputs           = tf.placeholder(tf.string, shape=[None, max_length])
	 
	# Inputs have (batch_size, timesteps) shape.
	inputs        = tf.placeholder(tf.int32, shape=[None, max_length])

	fine_labels   = tf.placeholder(tf.int64, shape=[None, max_length])

	instance_mask = tf.placeholder(tf.float32, shape=[None, max_length])
	
	# Keep_prob is a scalar.
	keep_prob     = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length    = tf.count_nonzero(fine_labels, axis=-1)
	# Create mask using sequence mask
	# pad_mask    = tf.sequence_mask(seq_length)
	pad_mask      = tf.to_float(tf.not_equal(fine_labels, 0))

	# Inf_mask
	fine_inf_mask = tf.placeholder(dtype=tf.float32, shape=[None, max_length, fine_classes])

	with tf.variable_scope("uni_embeddings"):
		embedding_matrix = tf.get_variable("uni_embeddings", shape=[vocab_size, embedding_size])
		embeddings       = tf.nn.embedding_lookup(embedding_matrix, inputs)
		# embeddings       = elmo_hub(inputs={"tokens": inputs, "sequence_len": seq_length}, signature="tokens", as_dict=True)["elmo"]

	with tf.variable_scope("rnn - encoder"):

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













def fine_attention_model(vocab_size, embedding_size, hidden_size, fine_classes, max_length):
	print("Creating TENSORFLOW model")
	elmo_hub = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
	inputs           = tf.placeholder(tf.string, shape=[None, max_length])
	 
	# Inputs have (batch_size, timesteps) shape.
	# inputs        = tf.placeholder(tf.int32, shape=[None, max_length])
	fine_labels   = tf.placeholder(tf.int64, shape=[None, max_length])

	instance_mask = tf.placeholder(tf.float32, shape=[None, max_length])
	
	# Keep_prob is a scalar.
	keep_prob     = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length    = tf.count_nonzero(fine_labels, axis=-1)
	# Create mask using sequence mask
	# pad_mask      = tf.sequence_mask(seq_length)
	pad_mask      = tf.to_float(tf.not_equal(fine_labels, 0))

	# Inf_mask
	fine_inf_mask = tf.placeholder(dtype=tf.float32, shape=[None, max_length, fine_classes])

	with tf.variable_scope("uni_embeddings"):
		# embedding_matrix = tf.get_variable("uni_embeddings", shape=[vocab_size, embedding_size])
		# embeddings       = tf.nn.embedding_lookup(embedding_matrix, inputs)
		embeddings       = elmo_hub(inputs={"tokens": inputs, "sequence_len": seq_length}, signature="tokens", as_dict=True)["elmo"]

	with tf.variable_scope("rnn"):

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
		
	
	with tf.variable_scope("dense"):
		logits = tf.layers.dense(self_att_concat, fine_classes, activation=None)

	with tf.variable_scope("infinite_mask"):
		inf_fine_logits = tf.math.add(logits, fine_inf_mask)

	with tf.variable_scope("loss"):
		padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fine_labels, logits=inf_fine_logits)
		padded_loss = tf.multiply(padded_loss, instance_mask)
		masked_loss = tf.boolean_mask(padded_loss, pad_mask)
		loss        = tf.reduce_mean(masked_loss)

	with tf.variable_scope("train"):
		train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
		# train_op    = tf.train.MomentumOptimizer(0.04, 0.95).minimize(loss)
	
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
















def fine_base_tf_model(vocab_size, embedding_size, hidden_size, fine_classes, max_length):
	print("Creating TENSORFLOW model")
	 
	# Inputs have (batch_size, timesteps) shape.
	inputs        = tf.placeholder(tf.int32, shape=[None, max_length])
	fine_labels   = tf.placeholder(tf.int64, shape=[None, max_length])

	instance_mask = tf.placeholder(tf.float32, shape=[None, max_length])
	
	# Keep_prob is a scalar.
	keep_prob     = tf.placeholder(tf.float32, shape=[])
	# Calculate sequence lengths to mask out the paddings later on.
	seq_length    = tf.count_nonzero(fine_labels, axis=-1)
	# Create mask using sequence mask
	pad_mask      = tf.to_float(tf.not_equal(fine_labels, 0))

	# Inf_mask
	fine_inf_mask = tf.placeholder(dtype=tf.float32, shape=[None, max_length, fine_classes])

	with tf.variable_scope("uni_embeddings"):
		embedding_matrix = tf.get_variable("uni_embeddings", shape=[vocab_size, embedding_size])
		embeddings       = tf.nn.embedding_lookup(embedding_matrix, inputs)

	with tf.variable_scope("rnn"):

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


	with tf.variable_scope("dense"):
		logits = tf.layers.dense(concat_out, fine_classes, activation=None)

	with tf.variable_scope("infinite_mask"):
		inf_fine_logits = tf.math.add(logits, fine_inf_mask)

	with tf.variable_scope("loss"):
		padded_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fine_labels, logits=inf_fine_logits)
		padded_loss = tf.multiply(padded_loss, instance_mask)
		masked_loss = tf.boolean_mask(padded_loss, pad_mask)
		loss        = tf.reduce_mean(masked_loss)

	with tf.variable_scope("train"):
		train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
		# train_op    = tf.train.MomentumOptimizer(0.04, 0.95).minimize(loss)

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




