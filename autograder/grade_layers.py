import layers
import layers_ref 

import numpy as np

np.set_printoptions(precision=4)
config =     {'FullyConnectedLayer':{'args':{'in_nodes':100,'out_nodes':10,'activation':'relu'},'shapes':[(16,100),(16,10),(16,100)],'fmarks':1,'bmarks':0.5,'umarks':0.5},
		   'ConvolutionLayer':{'args':{'in_channels':(3,32,32), 'filter_size':(2,2), 'numfilters':8, 'stride':2, 'activation':'relu'},'shapes':[(16,3,32,32),(16,8,16,16),(16,3,32,32)],'fmarks':2,'bmarks':1,'umarks':1},
		   'AvgPoolingLayer':{'args':{'in_channels':(3,32,32), 'filter_size':(2,2), 'stride':2},'shapes':[(16,3,32,32),(16,3,16,16),(16,3,32,32)],'fmarks':1.5,'bmarks':1.5,'umarks':0},
		   'MaxPoolingLayer':{'args':{'in_channels':(3,32,32), 'filter_size':(2,2), 'stride':2},'shapes':[(16,3,32,32),(16,3,16,16),(16,3,32,32)],'fmarks':1.5,'bmarks':1.5,'umarks':0},
		'FlattenLayer':{'args':{},'shapes':[(16,3,32,32),(16,3072),(16,3,32,32)],'fmarks':1,'bmarks':1,'umarks':0}
			}
def check_layer(layer_name):
	print("-------------------------")
	print("Grading {}".format(layer_name))
	marks = 0
	student_impl = getattr(layers,layer_name)
	ref_impl     = getattr(layers_ref,layer_name)

	try:
		student_layer = student_impl(**config[layer_name]['args'])
		ref_layer     = ref_impl(**config[layer_name]['args'])
	except Exception as e:
		print("Error in initializing {} - {}".format(layer_name,e))
		return marks

	try:
		ref_layer.weights = student_layer.weights.copy()
		ref_layer.biases  = student_layer.biases.copy()

	except:
		pass 

	X_shape, delta_shape, activation_shape = config[layer_name]['shapes']

	X     = np.random.normal(0.,5.,X_shape)
	delta = np.random.normal(0.,5.,delta_shape)
	act   = X.copy()

	# print(delta)
	try:
		if np.allclose(student_layer.forwardpass(X),ref_layer.forwardpass(X), rtol=0):
			marks += config[layer_name]['fmarks']
		else:
			print("Forward pass incorrect for {}".format(layer_name))
	except Exception as e:
		print("Error in forwardpass {} - {}".format(layer_name,e))
		return marks

	try:
		if np.allclose(student_layer.backwardpass(0.01,act,delta),ref_layer.backwardpass(0.01,act,delta), rtol=0):
			marks += config[layer_name]['bmarks']
		else:
			# print(X)
			# print(student_layer.forwardpass(X))
			# print("-----")
			# print(student_layer.backwardpass(0.01,act,delta))
			# print("------")
			# print(ref_layer.backwardpass(0.01,act,delta))
			# print("------")
			# print(ref_layer.backwardpass(0.01,act,delta) == student_layer.backwardpass(0.01,act,delta))
			print("Backwardpass incorrect for {}".format(layer_name))
	except Exception as e:
		print("Error in backwardpass {} - {}".format(layer_name,e))
		return marks

	try:
		if hasattr(ref_layer,'weights'):
			if np.allclose(ref_layer.weights,student_layer.weights) and np.allclose(ref_layer.biases,student_layer.biases):
				marks += config[layer_name]['umarks']
			else:
				ref_layer.weights = student_layer.weights.copy()
				ref_layer.biases  = student_layer.biases.copy()
				assert np.allclose(ref_layer.weights,student_layer.weights) and np.allclose(ref_layer.biases,student_layer.biases)
				ref_layer.forwardpass(X)    # This resets self.data
				student_layer.forwardpass(X)
				student_layer.backwardpass(0.16,act,delta)
				ref_layer.backwardpass(0.01,act,delta)   # In case someone scaled by number of examples
				if np.allclose(ref_layer.weights,student_layer.weights) and np.allclose(ref_layer.biases,student_layer.biases):
					marks += config[layer_name]['umarks']
				else:
					# print(ref_layer.weights-student_layer.weights)

					print("Weight update incorrect for {}".format(layer_name))
	except Exception as e:
		print("Error in backwardpass {} - {}".format(layer_name,e))
		return marks


	print("Marks obtained for {} is - {}".format(layer_name,marks))
	return marks



def grade_activation(func_name):

	print("-------------------------")
	print("Grading {}".format(func_name))
	marks = 0
	student_impl = getattr(layers,"{}_of_X".format(func_name))
	ref_impl     = getattr(layers_ref,"{}_of_X".format(func_name))

	student_impl_grad = getattr(layers,"gradient_{}_of_X".format(func_name))
	ref_impl_grad     = getattr(layers_ref,"gradient_{}_of_X".format(func_name))

	X       = np.random.normal(0.,1.,(16,10))
	delta_X = np.random.normal(0.,1.,(16,10))

	try:
		if np.allclose(student_impl(X),ref_impl(X), rtol=0):
			marks += 1
		else:
			print("Forward pass incorrect for {}".format(func_name))
	except Exception as e:
		print("Error in forwardpass {} - {}".format(func_name,e))
		return marks

	try:
		if np.allclose(student_impl_grad(X,delta_X),ref_impl_grad(X,delta_X), rtol=0):
			marks += 1
		else:
			if np.allclose(student_impl_grad(X,delta_X),ref_impl_grad(ref_impl(X),delta_X), rtol=0):
				# In case students recompute the activation of X
				marks += 1
			else:
				print("Backwardpass incorrect for {}".format(func_name))
	except Exception as e:
		print("Error in forwardpass {} - {}".format(func_name,e))
		return marks

	print("Marks obtained for {} is - {}".format(func_name,marks))

	return marks

def grade_layers():
	tot_marks = 0

	for f in ['relu','softmax']:
		tot_marks += grade_activation(f)


	for k in config.keys():
		tot_marks += check_layer(k)

	return tot_marks

if __name__ == '__main__':
	tot_marks = grade_layers()
	print("-----------------------")
	print("xxxxxxxxxxxxxxxxxxxxxxx")
	print("Total marks {}".format(tot_marks))
