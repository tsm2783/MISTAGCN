import node2vec
import networkx as nx
from gensim.models import Word2Vec
from pytictoc import TicToc
from mxnet import ndarray as nd

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

is_directed = False
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000

scenario = 'YellowTrip'  # Haikou, YellowTrip
adg_file_name = 'data/' + scenario + '/adg'
adg_tripplet_file_name = 'data/' + scenario + '/adg-tripplet.txt'
se_file_name = 'data/' + scenario + '/se.txt'

# generate tripplet form of adj

# N = len(ADG)
# for i in range(N):
#     for j in range(i,N):
#         if ADG[i,j] <= 0.1:
#             ADG[i,j] = 0
#             ADG[j,i] = 0

f = open(adg_tripplet_file_name, "w")
A, = nd.load(adg_file_name)
A = nd.sparse.csr_matrix(A)
cols = A.indices
data = A.data
rows = []
indptr = A.indptr
row = 0
for i in range(len(indptr) - 1):
    for j in range(indptr[i].asscalar(), indptr[i + 1].asscalar()):
        rows.append(row)
    row += 1

for i in range(len(rows)):
    f.write(f'{rows[i]} {cols[i].asscalar()} {data[i].asscalar()}\n')
f.close()

t.toc(f'{scenario}, finish generating tripplet form of ADG')


# generate se

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())
    return G


def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=1, sg=1,
        workers=14, epochs=iter)
    model.wv.save_word2vec_format(output_file)
    return


nx_G = read_graph(adg_tripplet_file_name)
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, se_file_name)

t.toc(f'{scenario}, finish generating se form of graph')
