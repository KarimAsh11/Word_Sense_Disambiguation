from tf_model_multi import multitask_seq_to_seq
from pred_utility   import load_parameters, write_pred, load_ckpt, pred_helper
import tensorflow   as tf



def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    #----Keep if required----
    tf.reset_default_graph()
    # -----------------------

    MAX_LENGTH, EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, DOM_CLASSES, LEX_CLASSES   = load_parameters(resources_path)
    test_corp_input, test_index_input, test_corp_instance, id_list, id_dict, inf_mask = pred_helper(input_path, resources_path, MAX_LENGTH, "fine")
    model_ckpt = load_ckpt(resources_path)

    inputs, fine_labels, dom_labels, lex_labels, fine_inf_mask, dom_inf_mask, lex_inf_mask, instance_mask, keep_prob, loss, train_op, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels = \
    multitask_seq_to_seq(EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, DOM_CLASSES, LEX_CLASSES, MAX_LENGTH)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        print("\nStarting prediction...")

        sess.run(tf.global_variables_initializer())
        sess.run(tf.initializers.local_variables())
        saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))

        fine_pred = sess.run([fine_predictions], feed_dict={inputs: test_corp_input, fine_labels: test_index_input, fine_inf_mask: inf_mask, keep_prob: 1})

    write_pred(fine_pred, id_list, id_dict, test_corp_instance, resources_path, output_path, "fine")



def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    #----Keep if required----
    tf.reset_default_graph()
    # -----------------------

    MAX_LENGTH, EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, DOM_CLASSES, LEX_CLASSES   = load_parameters(resources_path)
    test_corp_input, test_index_input, test_corp_instance, id_list, id_dict, inf_mask = pred_helper(input_path, resources_path, MAX_LENGTH, "dom")
    model_ckpt = load_ckpt(resources_path)

    inputs, fine_labels, dom_labels, lex_labels, fine_inf_mask, dom_inf_mask, lex_inf_mask, instance_mask, keep_prob, loss, train_op, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels = \
    multitask_seq_to_seq(EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, DOM_CLASSES, LEX_CLASSES, MAX_LENGTH)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        print("\nStarting prediction...")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initializers.local_variables())
        saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))

        dom_pred = sess.run([dom_predictions], feed_dict={inputs: test_corp_input, fine_labels: test_index_input, dom_inf_mask: inf_mask, keep_prob: 1})

    write_pred(dom_pred, id_list, id_dict, test_corp_instance, resources_path, output_path, "dom")



def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    #----Keep if required----
    tf.reset_default_graph()
    # -----------------------

    MAX_LENGTH, EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, DOM_CLASSES, LEX_CLASSES   = load_parameters(resources_path)
    test_corp_input, test_index_input, test_corp_instance, id_list, id_dict, inf_mask = pred_helper(input_path, resources_path, MAX_LENGTH, "lexname")
    model_ckpt = load_ckpt(resources_path)

    inputs, fine_labels, dom_labels, lex_labels, fine_inf_mask, dom_inf_mask, lex_inf_mask, instance_mask, keep_prob, loss, train_op, acc, fine_predictions, dom_predictions, lex_predictions, masked_fine_labels, masked_dom_labels, masked_lex_labels = \
    multitask_seq_to_seq(EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, DOM_CLASSES, LEX_CLASSES, MAX_LENGTH)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        print("\nStarting prediction...")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initializers.local_variables())
        saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))

        lex_pred = sess.run([lex_predictions], feed_dict={inputs: test_corp_input, fine_labels: test_index_input, lex_inf_mask: inf_mask, keep_prob: 1})

    write_pred(lex_pred, id_list, id_dict, test_corp_instance, resources_path, output_path, "lexname")

