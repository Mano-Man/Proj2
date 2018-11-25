# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
#                                                 Global Config
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def main():
	o = Optimizer(NeuralNetworkLoad(),test_set_size)
	best = o.by_max_granularity(worst_acc)
	print(f'Assimilated mask model with {best['acc']} test accuracy and {best['ops']} operations saved')

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class Optimizer:
    def __init__(self,net,test_set_size):
    	
    	self.mf= MaskFactory()
     	
    def by_max_granularity(self):
     	
     	# Create initial records
     	input_cfg = self.mf.createMaxGranInput()

     	pQ = PatchQuantizier(MAX_RECORDS) # Maybe it is better that PatchQuantizer will continue the chain of command instead of Optimizer.... 
     	output = pQ.simulate(input_cfg) 
     	pQ.save_state(PICKLE_FILE1) # We might want to load it again on errors
     	pQ = None # Might take a lot of memory - we might want to free it 

     	cQ = ChannelQuantizier(MAX_RECORDS2)
     	output = cQ.simulate(output)
     	cQ.save_state(PICKLE_FILE2)
     	pQ = None 
     	
     	lQ = LayerQuantizier(MAX_RECORDS3)
     	output = lQ.simulate(output)
     	lQ.save_state(PICKLE_FILE3)

     	return output 

    def by_uniform_filters(self): 
    	self.wQ = 

    def by_uniform_patches(self): 

    def by_uniform_layers(self): 



if __name__ == '__main__':
    main()


class PatchQuantizier:

	# l = Layer, c = Channel, p = Patch, m = Mask 
	# input: in[l][c][p][m] = (ops,acc,mask_id)
	#        mask_dict[mask_id] = mask  
	def __init__(self):
		pass

	def __str__(self):
		pass 
		# Defines a unique string that identifies the state, which will be saved to the pickle file 

	def simulate():
		pass 

