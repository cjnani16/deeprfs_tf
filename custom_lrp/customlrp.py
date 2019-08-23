#calculate EVERY LAYERS outputs
def run_all_layers(batch, model):

    outputs_ph = [layer.output for layer in model.layers[1:]]          # all layer outputs
    inputs_ph = [layer.input for layer in model.layers[1:]] 
    
    # Testing
    m = keras.models.Model(inputs=model2.input, outputs=outputs_ph)
    return [batch]+m.predict(batch)
    
outputs = run_all_layers(batch[0][np.newaxis,0,...], model2)

#prepare values needed for LRP
print(len(outputs),len(model2.layers))
weights = [layer.get_weights() for layer in model2.layers]
print(len(weights))
            

#turn values from arrays into dictionaries'
weights_dict = {}
outputs_dict = {}
inputs_dict = {}
layers_dict = {}
relevances_dict = {}
for i in range(len(outputs)):
    name = model2.layers[i].name
    outputs_dict[name] = outputs[i]
    weights_dict[name] = weights[i]
    if (type(model2.layers[i].input) != list):
        inputs_dict[name] = model2.layers[i].input.name.split("/")[0]
        if ("branch" in inputs_dict[name]):
            inputs_dict[name] = inputs_dict[name][0:-2]
    else:
        input_names = []
        for inp in model2.layers[i].input:
            n = inp.name.split("/")[0]
            if ("branch" in n):
                n = n[0:-2]
            input_names += [n]

        inputs_dict[name] = input_names

    layers_dict[name] = model2.layers[i]
    relevances_dict[name] = np.zeros_like(outputs[i])
    
#handy visualizer of graph
def print_shapes(model):
    for i in range(len(model.layers)):
        print("layer:",model.layers[i].name)
        if (len(weights_dict[model.layers[i].name])!=0):
            print('    has weights:')
            for w in weights_dict[model.layers[i].name]:
                print('    ',w.shape)
        else: print('    no weights')
    
        if (len(outputs_dict[model.layers[i].name])!=0):
            print('    has outputs:')
            print('    ',outputs_dict[model.layers[i].name].shape)
            
        if (type(inputs_dict[model.layers[i].name]) != list):
            print('    input name:', inputs_dict[model.layers[i].name])
        else:
            for n in range(len(inputs_dict[model.layers[i].name])):
                print('    input',n+1,'name:',(inputs_dict[model.layers[i].name][n]))

     
print_shapes(model2)


#create LRP backwards pass rule for every layer in this mf

#top dense layer
def LRPDense(inputs,weights,previous_relevances):
    relevances = np.zeros_like(inputs)
    #zijs
    preactivations = np.diag(inputs[0,:]).dot(weights[0])
    
    #split into positive and negative components
    preactivations_p = np.maximum(0,preactivations)
    preactivations_n = np.minimum(0,preactivations)
    preactivation_sums_p = np.sum(preactivations_p, axis=0) + np.maximum(0,weights[1])
    preactivation_sums_n = np.sum(preactivations_n, axis=0) + np.minimum(0,weights[1])
    
    preactivation_fraction_p = 0.5 * preactivations_p / preactivation_sums_p 
    preactivation_fraction_n = 0.5 * preactivations_n / preactivation_sums_n
    
    preactivation_fraction_p = np.where(np.isfinite(preactivation_fraction_p), preactivation_fraction_p, 0)
    preactivation_fraction_n = np.where(np.isfinite(preactivation_fraction_n), preactivation_fraction_n, 0)
    
    relevance_messages = (preactivation_fraction_n * previous_relevances) + (preactivation_fraction_p * previous_relevances)
    relevances = relevance_messages.sum(axis=1)
    return relevances

#global max-pooling layer
def LRPGlobMaxPool2D(inputs,previous_relevances):
    relevances = np.zeros_like(inputs)
    print(np.isfinite(previous_relevances).all())
    
    for m in range(len(inputs.max(axis=(0,1,2)))) :
        relevances[:,:,:,m] = np.where((inputs.max(axis=(0,1,2))[m]!=0) and inputs[:,:,:,m]==inputs.max(axis=(0,1,2))[m], 1, 0)
        #print('looking for ',inputs.max(axis=(0,1,2))[m],'in',inputs[:,:,:,m], relevances[:,:,:,m])
        relevances[:,:,:,m] *= previous_relevances[...,m]
    
    #print(np.isfinite(relevances).all())
    return relevances

#global average-pooling layer
def LRPGlobAvgPool2D(inputs,previous_relevances):
    relevances = np.zeros_like(inputs)
    print(np.isfinite(previous_relevances).all())
    
    for m in range(len(inputs.max(axis=(0,1,2)))) :
        if (np.sum(inputs[:,:,:,m])!=0):
            relevances[:,:,:,m] = previous_relevances[...,m]/np.sum(inputs[:,:,:,m])
        else:
            relevances[:,:,:,m] = 0

    return relevances

#simple activation layer
def LRPActivation(previous_relevances):
    return previous_relevances

#adding layer, need to branch the recursion here!
def LRPAdd(activations,inputs,weights,previous_relevances):
    #find the inputs being added
    print("    branching relevance to",inputs[0],"and",inputs[1])
    
    relevance_per_branch = []
    for input_name in inputs:
        relevance_per_branch += [previous_relevances/len(inputs)]
    
    return relevance_per_branch

#batchnorm layer
def LRPBatchNormV1(inputs,weights,previous_relevances):
        
    bn_gamma, bn_beta, bn_mean, bn_var = weights
    
    xhat_relevance = np.zeros_like(inputs)
    
    for channel in range(inputs.shape[3]):
        xhat_relevance[0,:,:,channel] = LRPDense(inputs[0,:,:,channel], [bn_gamma[channel],bn_beta[channel]], previous_relevances[0,:,:,channel])
    
    return xhat_relevance

def LRPResConv(inputs, kernels, previous_relevances):
    print("input:",inputs.shape)
    for kern in kernels:
        print('    kernel:',kern.shape)
    print(previous_relevances.shape)
    
    #get a preactivation for each kernel
    input_preactivations = np.zeros(inputs.shape + (kernels[0].shape[-1],))
    for i in range(kernels[0].shape[-1]):
        input_preactivations[...,i] = inputs * kernels[0][...,i]
    
    input_preactivations_p = np.maximum(0,input_preactivations)
    input_preactivations_n = np.minimum(0,input_preactivations)
    
    input_contributions_p = input_preactivations_p.sum(axis=-2) + np.maximum(0, kernels[1])
    input_contributions_n = input_preactivations_n.sum(axis=-2) + np.minimum(0, kernels[1])
    
    rel_fractions_p = input_preactivations_p / np.expand_dims(input_contributions_p, axis=3)
    rel_fractions_n = input_preactivations_n / np.expand_dims(input_contributions_n, axis=3)
    
    rel_fractions_p = np.where(np.isfinite(rel_fractions_p), rel_fractions_p, 0)
    rel_fractions_n = np.where(np.isfinite(rel_fractions_n), rel_fractions_n, 0)
    
    relevance_messages = .5 * rel_fractions_p * np.expand_dims(previous_relevances, axis=-2) + .5 * rel_fractions_n * np.expand_dims(previous_relevances, axis=-2)
    
    relevances = np.sum(relevance_messages, axis=-1)
    
    return relevances

def LRPGenConv(inputs, kernels, previous_relevances, activations, stride=1, padding=0):
    print("input:",inputs.shape)
    for kern in kernels:
        print('    kernel:',kern.shape)
    print(previous_relevances.shape)
    
    relevance = np.zeros_like(inputs)
    
    #relprop for each kernel to its RF
    for i in range(kernels[0].shape[-1]):
        for a in range(kernels[0].shape[0]):
            for b in range(kernels[0].shape[1]):
                trel = previous_relevances[0,a,b,i]
                c = (padding)+(stride)*(a)
                d = (padding)+(stride)*(b)
                rf = inputs[...,c:c+kernels[0].shape[0],d:d+kernels[0].shape[1],:]
                #print(rf.shape)
                
                preactivations = rf * kernels[0][:,:,:,i]
                preactivations_p = np.minimum(0,preactivations)
                preactivations_n = np.maximum(0,preactivations)
                
                preactivation_sum_p = np.sum(preactivations_p)
                preactivation_sum_n = np.sum(preactivations_n)
                
                rel_fractions_p = 0.5 * preactivations_p / preactivation_sum_p
                rel_fractions_n = 0.5 * preactivations_n / preactivation_sum_n
                
                rel_fractions_p = np.where(np.isfinite(rel_fractions_p), rel_fractions_p, 0)
                rel_fractions_n = np.where(np.isfinite(rel_fractions_n), rel_fractions_n, 0)
                
                relevance[...,c:c+kernels[0].shape[0],d:d+kernels[0].shape[1],:] += (trel * rel_fractions_p) + (trel * rel_fractions_n)
                
    return relevance
                                  
#Recursive LRP calculation starting at layer "name" and using the "relevances" dict
def LRP(name, relevances):
    print('Propagating relevance from \033[1m',name, "\033[0mto\033[1m", inputs_dict[name],"\033[0m")
    
    if ("dense" in name):
        print('    Detected FC layer...')
        relevances[inputs_dict[name]] += LRPDense(outputs_dict[inputs_dict[name]],weights_dict[name],relevances[name])
        
    elif ("max_pool" in name):
        print("    Detected max_pool layer...")
        relevances[inputs_dict[name]] += LRPGlobMaxPool2D(outputs_dict[inputs_dict[name]],relevances[name])
        
    elif ("global_average_pooling2d" in name):
        print("    Detected globalAVGpool layer...")
        relevances[inputs_dict[name]] += LRPGlobAvgPool2D(outputs_dict[inputs_dict[name]],relevances[name])
        
    elif ("activation" in name):
        print("    Detected activation layer...")
        relevances[inputs_dict[name]] += LRPActivation(relevances[name])
           
    elif ("add" in name):
        print("    Detected add layer...")
        branch_rels = LRPAdd(outputs_dict[name],inputs_dict[name],weights_dict[name],relevances[name])
        
        for i in range(len(inputs_dict[name])):
            relevances[inputs_dict[name][i]] += branch_rels[i]
    
    elif ("bn" in name):
        print("    Detected BatchNorm layer...")
        relevances[inputs_dict[name]] += LRPBatchNormV1(outputs_dict[inputs_dict[name]],weights_dict[name],relevances[name])
    
    elif ("res" in name):
        print("    Detected conv layer...")
        relevances[inputs_dict[name]] += LRPGenConv(outputs_dict[inputs_dict[name]],weights_dict[name],relevances[name], outputs_dict[name])
    
    else:
        print("!no LRP method known for",name,"!")
        return(relevances[name])
    
    if (type(inputs_dict[name]) != list):
        print('    Done. Total relevance sent:',np.sum(relevances[inputs_dict[name]]))
        print()
        return LRP(inputs_dict[name], relevances)
    else:
        print('    Done. Total relevance sent:',np.sum(relevances[inputs_dict[name][0]])+np.sum(relevances[inputs_dict[name][1]]))
        print()
        return (LRP(inputs_dict[name][0], relevances), LRP(inputs_dict[name][1], relevances))


#starting relevances
output_name = model2.layers[-1].name
first_rel = np.where(outputs[-1]==outputs[-1].max(), 1, 0)
relevances_dict[output_name] = first_rel
print(first_rel, np.argmax(first_rel))

print(sum(list(map(np.sum,relevances_dict.values()))))

LRP(output_name, relevances_dict)