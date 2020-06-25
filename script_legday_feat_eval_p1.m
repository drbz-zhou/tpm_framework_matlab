addpath(genpath('C:/Users/fredr/Documents/MATLAB/Add-Ons/ltfat'));
addpath(genpath('C:/Users/bzhou/Documents/MATLAB/ltfat'));
addpath(genpath('ltfat'));

% leg sampling rate 50 fps
streamsetting.win_size = 200; % 4 sec
streamsetting.win_step = 0.2*streamsetting.win_size;
streamsetting.win_mode = 0.2;
streamsetting.upscale  = 0; % the source data is already upscaled
streamsetting.adcbase  = 2^24;
%% time feats
for Person = 1%:6 %parfor here
    all_feats = [];
    all_label = [];
    for Recording = 1%:4
%         filename = ['D:\Pressure Sensing Data Archive\Leg-Day\dataset\balanced', ...
%             '\Person_',num2str(Person),'_Recording_',num2str(Recording),...
%             '_dataset_balanced.mat'];
%         
        filename = 'data/LegDay/Person_1_Recording_1_dataset_balanced.mat';
        balancedDataset = loadfeats(filename);
        disp(filename)
        for i=1:size(balancedDataset,1)
            stream=balancedDataset{i,1};
            % calculate temporal features from frame des
            tempfeats = stream2tempfeats(stream, streamsetting.win_size,...
                streamsetting.win_step, streamsetting.win_mode,...
                streamsetting.upscale, streamsetting.adcbase);
            
            
            all_feats = [all_feats, tempfeats];
            labels = ones(1,size(tempfeats,2))*balancedDataset{i,2};
            all_label = [all_label,labels];
            
            
            %if mod(i,50)==0
            disp(['Person ', num2str(Person),' ',num2str(i/size(balancedDataset,1))])
            %end
            % 
        end
        %clearvars c_streams
    end
    %savefeats(Person, all_feats, all_label, streamsetting);
end
%% spacefeats
for Person = 1%:6 %parfor here
    all_feats = [];
    all_label = [];
    for Recording = 1%:4
%         filename = ['D:\Pressure Sensing Data Archive\Leg-Day\dataset\balanced', ...
%             '\Person_',num2str(Person),'_Recording_',num2str(Recording),...
%             '_dataset_balanced.mat'];
        
        filename = 'data/LegDay/Person_1_Recording_1_dataset_balanced.mat';
        balancedDataset = loadfeats(filename);
        disp(filename)
        for i=1:size(balancedDataset,1)
            stream=balancedDataset{i,1};
            % calculate spacial features from key frames
            
            spacfeats = stream2spacfeats(stream, streamsetting.win_size,...
                streamsetting.win_step, ...
                streamsetting.upscale, streamsetting.adcbase);
            
            all_feats = [all_feats, spacfeats];
            labels = ones(1,size(spacfeats,2))*balancedDataset{i,2};
            all_label = [all_label,labels];
            
            
            %if mod(i,50)==0
            disp(['Person ', num2str(Person),' ',num2str(i/size(balancedDataset,1))])
            %end
            % 
        end
        %clearvars c_streams
    end
    %savefeats_s(Person, all_feats, all_label, streamsetting);
end
%%

function savefeats(Person, all_feats, all_label, streamsetting)
    save(['data\LegDay\LegDay_P', num2str(Person), '_tempfeats.mat'],...
        'all_feats','all_label','streamsetting','-v7.3');
end

function savefeats_s(Person, all_feats, all_label, streamsetting)
    save(['data\LegDay\LegDay_P', num2str(Person), '_spacfeats.mat'],...
        'all_feats','all_label','streamsetting','-v7.3');
end

function balancedDataset=loadfeats(filename)
        load(filename,'balancedDataset');
end

