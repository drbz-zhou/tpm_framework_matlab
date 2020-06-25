addpath(genpath('C:/Users/fredr/Documents/MATLAB/Add-Ons/ltfat'));
addpath(genpath('C:/Users/bzhou/Documents/MATLAB/ltfat'));

% the dining data has essentially 3 sensors, c_streams{1,1} is 3-by-1
% 40 fps
% define meta setting
streamsetting.win_size = 80;
streamsetting.win_step = 0.5*streamsetting.win_size;
streamsetting.win_mode = 0.2;
streamsetting.upscale  = 2;
streamsetting.adcbase  = 2^24;
%%

flag_t=0;
flag_s=1;
for Person = 1%:5%10
    all_feats = [];
    all_label = [];
    for Recording = 1%:8
%         filename = ['D:\Pressure Sensing Data Archive\Smart-Table\Thesis', ...
%             '\Person_',num2str(Person),'_Recording_',num2str(Recording),...
%             '.mat'];
        filename = 'data/table_v2/Person_1_Recording_1_dataset.mat';
        c_streams = loadstream(filename);
        disp(filename)
        for i=1:size(c_streams,1)
            patch_feats = [];
            % tablecloth has essentially 3 patches
            for Patch = 1:3
                stream=c_streams{i,1}{Patch,1};
                % calculate features from stream
                tempfeats = stream2tempfeats(stream, streamsetting.win_size,...
                    streamsetting.win_step, streamsetting.win_mode,...
                    streamsetting.upscale, streamsetting.adcbase);
                patch_feats = [patch_feats;tempfeats];
            end
            all_feats = [all_feats, patch_feats];
            labels = ones(1,size(patch_feats,2))*c_streams{i,2};
            all_label = [all_label,labels];
        end
        %clearvars c_streams
    end
    %savefeats(Person, all_feats, all_label, streamsetting);
end
%% spacefeats
spacfeats=[];
for Person = 1%:10 %parfor
    all_feats = [];
    all_label = [];
    for Recording = 1%:8
%         filename = ['H:\Pressure Sensing Data Archive\Smart-Table\Thesis', ...
%             '\Person_',num2str(Person),'_Recording_',num2str(Recording),...
%             '.mat'];
        filename = 'data/table_v2/Person_1_Recording_1_dataset.mat';
        c_streams = loadstream(filename);
        disp(filename)
        for i=1:size(c_streams,1)
            patch_feats = [];
            % tablecloth has essentially 3 patches
            for Patch = 1:3
                stream=c_streams{i,1}{Patch,1};
                % calculate features from stream
                spacfeats = stream2spacfeats(stream, streamsetting.win_size,...
                    streamsetting.win_step, ...
                    streamsetting.upscale, streamsetting.adcbase);
                patch_feats = [patch_feats;spacfeats];
            end
            all_feats = [all_feats, patch_feats];
            labels = ones(1,size(patch_feats,2))*c_streams{i,2};
            all_label = [all_label,labels];
            disp(['Person ', num2str(Person),' Recording ', num2str(Recording),' ',num2str(i/size(c_streams,1))])
            fv
        end
        %clearvars c_streams
    end
    %savefeats_s(Person, all_feats, all_label, streamsetting);
end
%%

function savefeats(Person, all_feats, all_label, streamsetting)
    save(['data\table_v2\Table_P', num2str(Person), '_tempfeats.mat'],...
        'all_feats','all_label','streamsetting','-v7.3');
end
function savefeats_s(Person, all_feats, all_label, streamsetting)
    save(['data\table_v2\Table_P', num2str(Person), '_spacfeats.mat'],...
        'all_feats','all_label','streamsetting','-v7.3');
end

function c_streams=loadstream(filename)
        load(filename,'c_streams');
end

