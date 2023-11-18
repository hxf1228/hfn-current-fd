clc;clear;close; % Release all data

%% Init configuration
fileLevel1 = ["N15_M01_F10","N09_M07_F10","N15_M07_F04","N15_M07_F10";"K1","K2","K3","K4"]'; % exp2
fileLevel2 = ["Healthy", "Inner", "Outer", 'Comb'];
rawDir = "/home/hxf/datasets/KAT/";
addpath(genpath(rawDir));

dataPoints = 4096;  % exp1 4096 exp2 8192 exp3 4096
shifSize = 1024; % exp1 1024 exp2 2048 exp3 1024
randomFlag = false;
method = "mvmd";  % vmd, mvmd, raw

% some sample parameters for MVMD and VMD
% 2000 1e-7
opt.alpha = 2000;        % moderate bandwidth constraint
opt.tau = 0;            % noise-tolerance (no strict fidelity enforcement)
opt.K = 5;              % 3 modes
opt.DC = 0;             % no DC part imposed
opt.init = 1;           % initialize omegas uniformly
opt.tol = 1e-7;

outPath = strcat("data/", method, "_test.mat");


%% Process funcion

for iLevel1 = 1:numel(fileLevel1(:,1))
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
                samples = 200;
                signalCut1 = zeros(samples,dataPoints);
                signalCut2 = zeros(samples,dataPoints);
                
                %  Random Sampling, Cut and normalize signal
                for iCut=1:samples
                    cutIndex = randomSerial(iCut);
                    signalCut1(iCut,:) = dataRaw((cutIndex+1):(cutIndex+dataPoints), 1);
                    signalCut2(iCut,:) = dataRaw((cutIndex+1):(cutIndex+dataPoints), 2);
                end  
    
                for iSignal=1:samples
                    iSignal
                    data_1 = signalCut1(iSignal,:);
                    data_2 = signalCut2(iSignal,:);
                    switch method
                        case "vmd"
                            [u_1, u_hat_1, omega_1] = vmd(data_1, opt);
                            [u_2, u_hat_2, omega_2] = vmd(data_2, opt);
                        case "mvmd"
                            data = [data_1;data_2];
                            [u, u_hat, omega] = mvmd_raw(data, opt); 
                            u_1 = u(1:3, :, 1);
                            u_2 = u(1:3, :, 2);
                        case "raw"
                            u_1 = data_1;
                            u_2 = data_2;
                    end
                    % feature caclulation
                    features_1 = featurecacul(u_1');
                    features_2 = featurecacul(u_2');
                    features_fusion = [features_1 features_2];
                    %features_fusion = features_2;
                    features = [features;features_fusion];
                    classes = [classes;iLevel2];
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
save(outPath, 'datamat', 'opt');

