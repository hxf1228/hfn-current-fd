clear all
clc
%rand('state', 2023);

load random.mat
%%
%load data1.mat

preprocess = "vmd";  % vmd, mvmd, raw
exp_index = 1;
iCondi = 2;
dataPath = strcat("data/", preprocess, "_", num2str(exp_index), ".mat");
%dataPath = strcat("data/", preprocess, "_", num2str(exp_index), ".mat");

save_path = strcat("output/", preprocess, "_1", "_cm.mat");

load(dataPath)

exp_num = 2;
%random_seed = randperm(2^16, exp_num) + 1;
%random_seed = [0, 9, 666, 700, 800, 1000, 2000, 2023, 2028, 5000];
results = [];
train_ratio = 0.9;

% 1:6 7:end raw 1:18 19:end vmd % 1:6 7:12 raw
features_raw = datamat(iCondi).features;
features = mapminmax(features_raw, 0, 1);
classes = datamat(iCondi).classes;

iSeed = 6

P_train = [];
T_train = [];
P_test = [];
T_test = [];
classe_num = classes(end,1);
each_class = length(classes)/classe_num;
train_num = round(train_ratio*each_class);
test_num = each_class-train_num;
train_i = train_index(iSeed,:) + 1;
test_i = test_index(iSeed,:) + 1;

P_train = features(train_i,:)';
T_train = classes(train_i,:)';
P_test = features(test_i,:)';
T_test = classes(test_i,:)';

[IW,B,LW,TF,TYPE] = elmtrain(P_train,T_train,250,'sig',1);

T_sim_1 = elmpredict(P_train,IW,B,LW,TF,TYPE);
T_sim_2 = elmpredict(P_test,IW,B,LW,TF,TYPE);

result_1 = [T_train' T_sim_1'];
result_2 = [T_test' T_sim_2'];

%%
k1 = length(find(T_train == T_sim_1));
n1 = length(T_train);
Accuracy_1 = k1 / n1 * 100;
results(iSeed, 1) = Accuracy_1;

%%
k2 = length(find(T_test == T_sim_2));
n2 = length(T_test);
Accuracy_2 = k2 / n2 * 100;
results(iSeed, 2) = Accuracy_2;
disp(['Testing acc = ' num2str(Accuracy_2) '%(' num2str(k2) '/' num2str(n2) ')'])

labels_test_all = T_test';
out_labels_test_all = T_sim_2';
cm = [labels_test_all out_labels_test_all];

save(save_path, "cm")
