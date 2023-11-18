clc;clear;close; % Release all data

%% Init configuration
fileLevel1 = ["N15_M01_F10","N09_M07_F10","N15_M07_F04","N15_M07_F10";"K1","K2","K3","K4"]'; % exp2
fileLevel2 = ["Healthy", "Inner", "Outer", 'Comb'];
rawDir = "/home/hxf/datasets/KAT/";
addpath(genpath(rawDir));

dataPoints = 4096;  % exp1 4096 exp2 8192 exp3 4096
shifSize = 1024; % exp1 1024 exp2 2048 exp3 1024
randomFlag = false;
method = "raw";  % vmd, mvmd, raw

% some sample parameters for MVMD and VMD
alpha = 2000;        % exp1 2000 3 exp2 10000 3
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
K = 5;              % 3 modes
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 1e-7;

outPath = strcat("data/", method, "_test.mat");

%p = parpool('local',8);
%% Process funcion

for iLevel1 = 1:numel(fileLevel1(:,1))
    ifeatures = 1;
    features = [];
    classes = [];
    conditionName = fileLevel1(iLevel1,1);
    conditionNum = fileLevel1(iLevel1,2);
    conditionNum
    level1Dir = strcat(rawDir, fileLevel1(iLevel1), '/');
    for iLevel2 = 1:numel(fileLevel2)
        iLevel2
        level2Dir = strcat(level1Dir, fileLevel2(iLevel2));
        level2Files = dir(strcat(level2Dir,"/*.csv"));
        lv2TimeSum = 0;
        switch iLevel2
            case 1
                faultsName = "K001";
            case 2
                faultsName = "KA04";
            case 3
                faultsName = "KI04";
            case 4
                faultsName = "KB23";
        end
        for iMatfiles = 1: numel(faultsName)
            fault_name = faultsName(iMatfiles);
            %--------------------------------------------------------------
            %  Read Mat File
            %--------------------------------------------------------------
            for iExp = 1: 5
                mat_name = strcat(conditionName,"_",fault_name,"_", num2str(iExp)); 
                mat_path = strcat(rawDir,fault_name,'/',mat_name,'.mat');
                load(mat_path);
                mat_variable = eval(mat_name);
                length4Khz = 16000;
                %length64Khz = length4Khz*16;
                length64Khz = 250000;
                signal_current_1_raw = (mat_variable.Y(2).Data)'; 
                signal_current_2_raw = (mat_variable.Y(3).Data)';
                signal_vibration_raw = (mat_variable.Y(7).Data)';
                
                signal_current_1 = signal_current_1_raw(1:length64Khz,:);
                signal_current_2 = signal_current_2_raw(1:length64Khz,:);
                signal_vibration = signal_vibration_raw(1:length64Khz,:);
                
                dataRaw = [signal_current_1 signal_current_2];
                
                
                starttime = tic;
                if randomFlag==false
                    randomSerial = 1:shifSize:length(dataRaw)-dataPoints;
                    randomFlag=true;
                end
                [~,samples] = size(randomSerial);
                samples = 20;
                signalCut1 = zeros(samples,dataPoints);
                signalCut2 = zeros(samples,dataPoints);
                tic;
                %  Random Sampling, Cut and normalize signal
                for iCut=1:samples
                    cutIndex = randomSerial(iCut);
                    signalCut1(iCut,:) = dataRaw((cutIndex+1):(cutIndex+dataPoints), 1);
                    signalCut2(iCut,:) = dataRaw((cutIndex+1):(cutIndex+dataPoints), 2);
                end  
    
                for iSignal=1:samples
                    iSignal
                    data_1_cut = signalCut1(iSignal,:);
                    data_2_cut = signalCut2(iSignal,:);
%                     data_1_fft = fft(data_1_cut);
%                     data_1 = abs(data_1_fft);
%                     data_2_fft = fft(data_2_cut);
%                     data_2 = abs(data_2_fft);
                    data_1 = data_1_cut;
                    data_2 = data_2_cut;
                    %data_1 = normalize1(data_1);
                    %data_2 = normalize1(data_2);

                    data = [data_1;data_2];
                    feature = data;
%                     a = u_1_reshape(:,:,1);
%                     a_1 = reshape(a, [1, 4096]);
                    features(ifeatures,:,:,:,:) = feature;
                    classes = [classes;iLevel2];
                    ifeatures = ifeatures+1;
                end
                endtime = toc(starttime);
                timeper = endtime /samples * 1000
            end
        end

    end
    datamat(iLevel1).condition = conditionNum;
    datamat(iLevel1).features = features;
    datamat(iLevel1).classes = classes;
    clear features
    clear classes
end
%delete(p);

save(outPath, 'datamat');

