import scipy.io as scio

datapath = '../data/mvmdf_1.mat'
matdata = scio.loadmat(datapath)

data = matdata['datamat'][0, 0]['features']
data_reshape = data.reshape((-1, 3, 64, 64, 2))
labels = matdata['datamat'][0, 0]['classes']
print("test")
