from parse_utility import get_syns, get_wn_syns, get_map_dict, get_map_dict_tsv, get_map_dict_tsv_inv




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




resources_path 	 		 = 	"../resources"
babel_word_path		     = resources_path+"/babelnet2wordnet.tsv"
babel_dom_path			 = resources_path+"/babelnet2wndomains.tsv"
babel_lex_path		     = resources_path+"/babelnet2lexnames.tsv"
word_to_babel_map        = get_map_dict_tsv_inv(babel_word_path)
babel_to_dom_map         = get_map_dict_tsv(babel_dom_path)
babel_to_lex_map 	     = get_map_dict_tsv(babel_lex_path)

#Adjust dataset path accordingly
dataset_path		     = resources_path+"/datasets/senseval3.gold.key.txt"
fine_sense_map           = get_map_dict(dataset_path)

id_list  = list(fine_sense_map.keys())
id_list.sort()

fine_list = []
dom_list = []
lex_list = []

for i in id_list:
	wn = get_wn_syns(fine_sense_map[i])

	fine_list.append(word_to_babel_map[wn])
	if babel_to_dom_map.get(word_to_babel_map[wn]) is not None:	
		dom_list.append(babel_to_dom_map[word_to_babel_map[wn]])
	else:
		dom_list.append("factotum")

	lex_list.append(babel_to_lex_map[word_to_babel_map[wn]])


write_list(fine_list, id_list, resources_path+"/gold_fine.txt")
write_list(dom_list, id_list, resources_path+"/gold_dom.txt")
write_list(lex_list, id_list, resources_path+"/gold_lex.txt")

fine_sense_path		     = resources_path+"/gold_fine.txt"
dom_sense_path		     = resources_path+"/gold_dom.txt"
lex_sense_path		     = resources_path+"/gold_lex.txt"
fine_sense_map           = get_map_dict(fine_sense_path)
dom_sense_map            = get_map_dict(dom_sense_path)
lex_sense_map            = get_map_dict(lex_sense_path)

fine_pred_path		     = resources_path+"/final_fine.txt"
dom_pred_path		     = resources_path+"/final_dom.txt"
lex_pred_path		     = resources_path+"/final_lex.txt"
fine_pred_map            = get_map_dict(fine_pred_path)
dom_pred_map             = get_map_dict(dom_pred_path)
lex_pred_map             = get_map_dict(lex_pred_path)

f1_fine = 0
f1_dom = 0
f1_lex = 0
for x in fine_sense_map:
	if fine_sense_map[x] == fine_pred_map[x]:
		f1_fine+=1
	if dom_sense_map[x] == dom_pred_map[x]:
		f1_dom+=1
	if lex_sense_map[x] == lex_pred_map[x]:
		f1_lex+=1


print("f1 fine:", f1_fine/len(fine_sense_map))


print("f1 dom:", f1_dom/len(fine_sense_map))
print("f1 lex:", f1_lex/len(lex_pred_map))

print(len(fine_sense_map))
print(len(fine_pred_map))
print(len(dom_sense_map))
print(len(dom_pred_map))
print(len(lex_sense_map))
print(len(lex_pred_map))









