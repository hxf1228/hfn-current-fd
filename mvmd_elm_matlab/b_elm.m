clear all
clc
%rand('state', 2023);

load random.mat

%load data1.mat

preprocess = "vmd";  % vmd, mvmd, raw
exp_index = 1;
%dataPath = strcat("data/", preprocess, "_p", num2str(exp_index), ".mat");
dataPath = strcat("data/", preprocess, "_", num2str(exp_index), ".mat");
%dataPath = strcat("data/", preprocess, "_hc", num2str(exp_index), ".mat");

load(dataPath)

exp_num = 10;
%random_seed = randperm(2^16, exp_num) + 1;
%random_seed = [0, 9, 666, 700, 800, 1000, 2000, 2023, 2028, 5000];
results = [];
train_ratio = 0.9;

for iCondi = 1:1:4
    % 1:6 7:end raw 1:18 19:end vmd
    features_raw = datamat(iCondi).features; % 1:6 7:12 raw
    features = mapminmax(features_raw, 0, 1);
    classes = datamat(iCondi).classes;

    for iSeed = 1:1:exp_num

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

        %rand('state', iSeed)

        [IW,B,LW,TF,TYPE] = elmtrain(P_train,T_train,250,'sig',1); %250
        T_sim_1 = elmpredict(P_train,IW,B,LW,TF,TYPE);

        starttime = tic;
        T_sim_2 = elmpredict(P_test(:,1),IW,B,LW,TF,TYPE);
        endtime = toc(starttime);
        timeper = endtime * 1000

        result_1 = [T_train' T_sim_1'];
        result_2 = [T_test' T_sim_2'];
        k1 = length(find(T_train == T_sim_1));
        n1 = length(T_train);
        Accuracy_1 = k1 / n1 * 100;
        results(iSeed, 1) = Accuracy_1;

        %%
        k2 = length(find(T_test == T_sim_2));
        n2 = length(T_test);
        Accuracy_2 = k2 / n2 * 100;
        results(iSeed, 2) = Accuracy_2;
   
    end
    train_mean_acc = mean(results(:,1));
    train_std_acc = std(results(:,1));
    test_mean_acc = mean(results(:,2));
    test_std_acc = std(results(:,2));
    test_sort = sort(results(:,2));
    best_test = test_sort(1:10);
    best_mean = mean(best_test);
    best_std = std(best_test);

    disp(['Condition ' datamat(iCondi).condition])
    disp(['Traing Mean ' num2str(train_mean_acc) ' Std ' num2str(train_std_acc)])
    disp(['Testing Mean ' num2str(test_mean_acc) ' Std ' num2str(test_std_acc)])
    disp(['Testing BMean ' num2str(best_mean) ' BStd ' num2str(best_std)])

end

