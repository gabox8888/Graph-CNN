from graphcnn.experiment import *
import pickle

root = "./datasets/amr/"

adj = np.load(root + 'adjacency.npy')  
vertex = np.load(root + 'vertex.npy')  
labels = np.load(root + 'labels.npy')  
masks = np.load(root + 'masks.npy')  
i_word = pickle.load(open(root + 'amr_i_word.pkl','rb'))
dataset = [adj,vertex,labels,masks,i_word]

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

            
        net.make_rnn_layer(300,200,6004)
        
exp = GraphCNNWithRNNExperiment('AMR', 'amr', AMRExperiment())

exp.num_iterations = 1500
exp.train_batch_size = 70
exp.test_batch_size = 30
exp.optimizer = 'adam'
exp.debug = True

exp.preprocess_data(dataset)
acc, std = exp.run_kfold_experiments(no_folds=10)
print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))