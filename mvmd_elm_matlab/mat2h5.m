condition_1 = datamat(1).features;
condition_2 = datamat(2).features;
condition_3 = datamat(3).features;
condition_4 = datamat(4).features;
label_1 = datamat(1).classes;
label_2 = datamat(2).classes;
label_3 = datamat(3).classes;
label_4 = datamat(4).classes;

data_name = 'mvmd.h5';
datatype = 'uint8';
chunksize = [1, 2, 64, 64];
data_size = size(condition_1);
label_size = size(label_1);
h5create(data_name, '/K1', data_size, ...
    'ChunkSize',chunksize,'Datatype',datatype,'Deflate',9);
h5create(data_name, '/K2', data_size, ...
    'ChunkSize',chunksize,'Datatype',datatype,'Deflate',9);
h5create(data_name, '/K3', data_size, ...
    'ChunkSize',chunksize,'Datatype',datatype,'Deflate',9);
h5create(data_name, '/K4', data_size, ...
    'ChunkSize',chunksize,'Datatype',datatype,'Deflate',9);

h5create(data_name, '/K1Label', label_size, ...
    'ChunkSize',[1 1],'Datatype','int32','Deflate',9);
h5create(data_name, '/K2Label', label_size, ...
    'ChunkSize',[1 1],'Datatype','int32','Deflate',9);
h5create(data_name, '/K3Label', label_size, ...
    'ChunkSize',[1 1],'Datatype','int32','Deflate',9);
h5create(data_name, '/K4Label', label_size, ...
    'ChunkSize',[1 1],'Datatype','int32','Deflate',9);

h5write(data_name, '/K1', condition_1)
h5write(data_name, '/K2', condition_2)
h5write(data_name, '/K3', condition_3)
h5write(data_name, '/K4', condition_4)

h5write(data_name, '/K1Label', label_1)
h5write(data_name, '/K2Label', label_2)
h5write(data_name, '/K3Label', label_3)
h5write(data_name, '/K4Label', label_4)