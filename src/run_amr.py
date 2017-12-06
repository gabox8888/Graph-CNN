from graphcnn.experiment import *
import pickle

root = "./datasets/amr/"

adj = []
vertex = []
labels = []
masks = []
linear = []
linear_masks = [] 

for i in range(10312,20000):
    if i % 1000 == 0 : print("currently at {}".format(i))
    adj += [np.load(root + 'vertex_{}.npy'.format(i))]
    vertex += [np.load(root + 'adjacency_{}.npy'.format(i))]  
    labels += [np.load(root + 'labels_{}.npy'.format(i))] 
    masks += [np.load(root + 'masks_{}.npy'.format(i))]  
    linear += [np.load(root + 'linear_{}.npy'.format(i))]  
    linear_masks += [np.load(root + 'masks_linear_{}.npy'.format(i))]  

i_word = pickle.load(open(root + 'amr_i_word.pkl','rb'))

adj = np.array(adj)
vertex = np.array(vertex)
labels = np.array(labels)
masks = np.array(masks)
linear = np.array(linear)
linear_masks = np.array(linear_masks)
dataset = [adj,vertex,labels,masks,linear,linear_masks,i_word]


# Decay value for BatchNorm layers, seems to work better with 0.3
GraphCNNGlobal.BN_DECAY = 0.3
class AMRExperiment(object):
    def create_network(self, net, input):
        net.create_network(input)
        net.make_graphcnn_layer(64)
        net.make_graphcnn_layer(64)
        net.make_graph_embed_pooling(no_vertices=32)
            
        net.make_graphcnn_layer(32)
        
        net.make_graph_embed_pooling(no_vertices=8)
        net.make_fc_layer(300)

            
        net.make_rnn_layer(300,200,6504)
        
exp = GraphCNNWithRNNExperiment('AMR', 'amr', AMRExperiment())

exp.num_iterations = 1500
exp.train_batch_size = 70
exp.test_batch_size = 30
exp.optimizer = 'adam'
exp.debug = True

exp.preprocess_data(dataset)
# exp.min_num_file = 11000
# exp.max_num_file = 11500
# exp.root_dir = root
# exp.i_to_word = i_word
# exp.no_samples = 500
acc, std = exp.run_kfold_experiments(no_folds=10)
print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))