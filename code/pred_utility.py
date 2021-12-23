from utility import load_pickle, save_pickle, encode_input_to_id, encode_instance_mask, encode_elmo
from pre_tf import prepare_mask_infinity
from parse_utility import get_map_dict_tsv, get_map_dict_tsv_inv
from lxml import etree
from nltk.corpus import wordnet as wn

def load_parameters(resources_path):
	"""
	Load hyperparameters.

	params:

	returns: int, int, int
	"""
	MAX_LENGTH             = 60
	EMBEDDING_SIZE   	   = 200
	HIDDEN_SIZE      	   = 256
	
	fine_senses_id_dict	   = load_pickle(resources_path+"/fine_senses_id_dict.pickle")
	coarse_domain_id_dict  = load_pickle(resources_path+"/coarse_domain_id_dict.pickle")
	coarse_lex_id_dict     = load_pickle(resources_path+"/coarse_lex_id_dict.pickle")
	
	FINE_CLASSES     	   = len(fine_senses_id_dict)
	DOM_CLASSES       	   = len(coarse_domain_id_dict)
	LEX_CLASSES            = len(coarse_lex_id_dict)
	
	return MAX_LENGTH, EMBEDDING_SIZE, HIDDEN_SIZE, FINE_CLASSES, DOM_CLASSES, LEX_CLASSES

def get_pos():
	"""
	possible pos to be passed to mfs.

	params:
	returns: dict	
	"""	
	pos_dict = dict()
	pos_dict["noun"] = wn.NOUN
	pos_dict["verb"] = wn.VERB
	pos_dict["adj"]  = wn.ADJ
	pos_dict["adv"]  = wn.ADV

	return pos_dict

def mf_sense(lemma_pos, resources_path, mode):
	"""
	Returns most frequent sense for this lemma pos pair .

	params:
		lemma_pos (str)
		resources_path (str)
		mode (str)

    returns: str	
	"""

	babel_word_path		     = resources_path+"/babelnet2wordnet.tsv"
	babel_dom_path			 = resources_path+"/babelnet2wndomains.tsv"
	babel_lex_path		     = resources_path+"/babelnet2lexnames.tsv"
	word_to_babel_map        = get_map_dict_tsv_inv(babel_word_path)
	babel_to_dom_map         = get_map_dict_tsv(babel_dom_path)
	babel_to_lex_map 	     = get_map_dict_tsv(babel_lex_path)

	wn_mfs = wn_mf_sense(lemma_pos)

	if word_to_babel_map.get(wn_mfs) is not None:
		bn_mfs = word_to_babel_map[wn_mfs]
	else:
		bn_mfs = "<UNK>"
		# print("not in mapping", wn_mfs)

	if mode == "fine":
		mfs   = bn_mfs
	elif mode == "dom":
		if babel_to_dom_map.get(bn_mfs) is not None:
			mfs  = babel_to_dom_map[bn_mfs]
		else:
			mfs = "factotum"
	elif mode == "lexname":
		if babel_to_lex_map.get(bn_mfs) is not None:
			mfs  = babel_to_lex_map[bn_mfs]
		else:
			mfs = "<UNK>"
	else:
		print("Please specify prediction mode")
		exit()

	return mfs

def wn_mf_sense(lemma_pos):
	"""
	Returns most frequent sense for this lemma pos pair - wordnet.

	params:
		lemma_pos (str)

	returns: str	
	"""
	pos_dict    = get_pos()
	lemma, pos  = lemma_pos.rsplit("_", 1)

	wn_syns     = wn.synsets(lemma, pos=pos_dict.get(pos, None))
	if len(wn_syns) == 0:
		mfs = "<UNK>"
		print("not in wn")

	else:
		mfs = wn_syns[0]

	return 'wn:' + str(mfs.offset()).zfill(8) + mfs.pos()



def instance_flatten(inst):
	"""
	reshape instance matrix.

	params:
		inst (list)

	returns: list	
	"""
	flat_inst = []
	for i in inst:
		for j in i:
			flat_inst.append(j)
	return flat_inst




def write_pred(pred, id_list, id_dict, test_corp_instance, resources_path, output_path, mode):
	"""
	prepare prediction txt.

	params:
		pred (np array)
		id_list (list)
		id_dict (dict)
		test_corp_instance (list)
		resources_path (str)
		output_path (str)
		mode (str)
	"""
	if mode == "fine":
		sense_id_dict  = load_pickle(resources_path+"/id_to_fine_senses.pickle")
	elif mode == "dom":
		sense_id_dict  = load_pickle(resources_path+"/id_to_coarse_domain.pickle")
	elif mode == "lexname":
		sense_id_dict  = load_pickle(resources_path+"/id_to_coarse_lex.pickle")
	else:
		print("Please specify prediction mode")
		exit()

	senses = []
	pred = pred[0].tolist()
	instance_list = instance_flatten(test_corp_instance)
	id_iterator = 0
	for i, j in enumerate(instance_list):
		if j == 1:
			if pred[i] == 1:
				lemma_pos = id_dict[id_list[id_iterator]]
				sense     = mf_sense(lemma_pos, resources_path, mode)
				senses.append(sense)
				id_iterator +=1

			elif pred[i] == 0 or pred[i] == 2:
				print("troubleshoot")

			else:
				sense = sense_id_dict[pred[i]]
				senses.append(sense)
				id_iterator +=1

	write_list(senses, id_list, output_path)


def write_list(senses, id_list, output_path):
	"""
	write prediction txt.

	params:
		pred_dict (dict)
		id_list (list)
		output_path (str)
	"""
	# print(senses)
	with open(output_path, 'w') as f:
		for i, j in enumerate(id_list):
			f.write(j + " " + senses[i])
			f.write('\n')


def write(pred_dict, id_list, output_path):
	"""
	write prediction txt.

	params:
		pred_dict (dict)
		id_list (list)
		output_path (str)
	"""
	# print(pred_dict)
	with open(output_path, 'w') as f:
		for i in id_list:
			f.write(i + " " + pred_dict[i])
			f.write('\n')


def load_ckpt(resources_path):
	"""
	loads checkpoint path for best model.

	params:
	    resources_path (str)
	returns: str
	"""
	ckpt = resources_path+"/best_model"
	return ckpt


def prep_to_class(test_corp_token, test_corp_lemma_pos, test_corp_instance, max_length):
	"""
	Prepare input for classification.

	params:
		test_corp_token (list)
		test_corp_lemma_pos (list)
		test_corp_instance (list)
		max_length (int)
	returns: list, list, list, list
	"""

	for i, line in enumerate(test_corp_token):
		if len(line)>max_length: 
			test_corp_token.insert(i+1, test_corp_token[i][max_length:])
			test_corp_token[i]=test_corp_token[i][:max_length]
			
			test_corp_lemma_pos.insert(i+1, test_corp_lemma_pos[i][max_length:])
			test_corp_lemma_pos[i]=test_corp_lemma_pos[i][:max_length]
			
			test_corp_instance.insert(i+1, test_corp_instance[i][max_length:])
			test_corp_instance[i]=test_corp_instance[i][:max_length]

	return test_corp_token, test_corp_lemma_pos, test_corp_instance



def test_parse(input_path):
	"""
	Parses test dataset.

	params:
	    input_path	(str)
	returns: list, list, list, list, list, dict
	"""
	i=0
	test_corp_token       = []
	test_corp_lemma_pos   = []
	test_corp_pos         = []
	test_corp_instance    = []

	id_list				  = []
	test_id				  = []
	id_dict				  = dict()
	# id : lemma_pos

	print("Parsing XML file ...")

	for event, element in etree.iterparse(input_path, tag="sentence"):
		token_sentence          = []
		lemma_pos_sentence      = []
		pos_sentence            = []
		instance_sentence       = []

		i+=1

		if event == "end":
			for el in element.iter():
				if el.text is not None:
					if el.tag == "wf":
						token = el.text.lower().replace(" ", "_")
						lemma = el.attrib["lemma"].lower().replace(" ", "_")
						pos   = el.attrib["pos"].lower()
						lemma_pos = '_'.join([lemma, pos])

						token_sentence.append(token)
						lemma_pos_sentence.append(lemma_pos)
						pos_sentence.append(pos)
						instance_sentence.append(0)
						test_id.append(0)


					if el.tag == "instance":
						token = el.text.lower().replace(" ", "_")
						lemma = el.attrib["lemma"].lower()
						pos   = el.attrib["pos"].lower()

						lemma_pos = '_'.join([lemma, pos])

						if el.attrib['id']:
							id_list.append(el.attrib['id'])
							test_id.append(el.attrib['id'])
							id_dict[el.attrib['id']] = lemma_pos

							token_sentence.append(token)
							lemma_pos_sentence.append(lemma_pos)
							pos_sentence.append(pos)
							instance_sentence.append(1)

		
						else:
							print("------------------------------------------------------ ID MISSING ------------------------------------------------------")

		test_corp_token.append(token_sentence)
		test_corp_lemma_pos.append(lemma_pos_sentence)
		test_corp_pos.append(pos_sentence)
		test_corp_instance.append(instance_sentence)


	return test_corp_token, test_corp_lemma_pos, test_corp_instance, id_list, id_dict, test_id



def pred_helper(input_path, resources_path, MAX_LENGTH, mode):

	"""
	Prepares for the prediction.

	params:
		input_path (str)
		resources_path (str)
		MAX_LENGTH (int)
		mode (str)

	returns: np array, np array, list, list, dict, np array
	"""
	test_corp_token, test_corp_lemma_pos, test_corp_instance, id_list, id_dict, test_id = test_parse(input_path)
	test_corp_token, test_corp_lemma_pos, test_corp_instance = prep_to_class(test_corp_token, test_corp_lemma_pos, test_corp_instance, MAX_LENGTH)

	if mode == "fine":
		sense_id_dict   = load_pickle(resources_path+"/fine_senses_id_dict.pickle")
		inf_dict        = load_pickle(resources_path+"/all_id_cand_fine_senses.pickle")
	elif mode == "dom":
		sense_id_dict   = load_pickle(resources_path+"/coarse_domain_id_dict.pickle")
		inf_dict        = load_pickle(resources_path+"/all_id_cand_coarse_domain.pickle")
	elif mode == "lexname":
		sense_id_dict   = load_pickle(resources_path+"/coarse_lex_id_dict.pickle")
		inf_dict        = load_pickle(resources_path+"/all_id_cand_coarse_lex.pickle")
	else:
		print("Please specify prediction mode")

	lemma_pos_id_dict   = load_pickle(resources_path+"/lemma_pos_id_dict.pickle")
	NUM_CLASSES         = len(sense_id_dict)
	
	test_elmo_input     = encode_elmo(test_corp_token, MAX_LENGTH)	
	test_index_input    = encode_input_to_id(test_corp_lemma_pos, lemma_pos_id_dict, MAX_LENGTH)
	test_index_instance = encode_instance_mask(test_corp_instance, MAX_LENGTH)		

	inf_fine_mask       = prepare_mask_infinity(test_index_input, test_index_instance, inf_dict, MAX_LENGTH, NUM_CLASSES)

	return test_elmo_input, test_index_input, test_corp_instance, id_list, id_dict, inf_fine_mask

