addpath(genpath('C:/Users/fredr/Documents/MATLAB/Add-Ons/ltfat'));
addpath(genpath('C:/Users/bzhou/Documents/MATLAB/ltfat'));

datapath = 'C:\owncloud\Work\Projects\Robot Skin\matlab_program\Data\Samples';

streamsetting.win_mode = 0.2;
streamsetting.upscale  = 2; 
streamsetting.adcbase  = 2^12;

parfor Person = 6:29
    all_feats = [];
    all_label = [];
    for Recording = 1:2
        filename=[datapath,'\dataset',num2str(Person),'_',num2str(Recording)];
        disp(filename);
        vNewDataset=loaddata(filename);
        for vClass=1:length(vNewDataset)
            for vSample = 1:length(vNewDataset{vClass})
                sampledata = vNewDataset{vClass}{vSample};
                tempfeats = stream2tempfeats(sampledata, size(sampledata,3),...
                    0, streamsetting.win_mode, ...
                    streamsetting.upscale, streamsetting.adcbase);
                all_feats = [all_feats, tempfeats];
                labels = ones(1,size(tempfeats,2))*vClass;
                all_label = [all_label,labels];
            end
        end
    end
    savefeats(Person, all_feats, all_label, streamsetting);
end
parfor Person = 6:29
    all_feats = [];
    all_label = [];
    for Recording = 1:2
        filename=[datapath,'\dataset',num2str(Person),'_',num2str(Recording)];
        disp(filename);
        vNewDataset=loaddata(filename);
        for vClass=1:length(vNewDataset)
            for vSample = 1:length(vNewDataset{vClass})
                sampledata = vNewDataset{vClass}{vSample};
                spacfeats = stream2spacfeats(sampledata, size(sampledata,3),...
                    0, ...
                    streamsetting.upscale, streamsetting.adcbase);
                all_feats = [all_feats, spacfeats];
                labels = ones(1,size(spacfeats,2))*vClass;
                all_label = [all_label,labels];
            end
        end
    end
    savefeats_s(Person, all_feats, all_label, streamsetting);
end
%% put all data together
data_all = [];
for Person = 6:29
    filename=['data\RoboSkin\RoboSkin_P', num2str(Person), '_tempfeats.mat'];
    load(filename)
    all_feats_t=all_feats;
    filename=['data\RoboSkin\RoboSkin_P', num2str(Person), '_spacfeats.mat'];
    load(filename)
    all_feats_s=all_feats;
    all_feats=[all_feats_t;all_feats_s];
    
    X=all_feats';
    y=all_label';
    X( isnan(X) ) = 0;
    data=[X,y];
    data_all=[data_all;data];
end
%%
function savefeats(Person, all_feats, all_label, streamsetting)
    save(['data\RoboSkin\RoboSkin_P', num2str(Person), '_tempfeats.mat'],...
        'all_feats','all_label','streamsetting','-v7.3');
end
function savefeats_s(Person, all_feats, all_label, streamsetting)
    save(['data\RoboSkin\RoboSkin_P', num2str(Person), '_spacfeats.mat'],...
        'all_feats','all_label','streamsetting','-v7.3');
end
function vNewDataset=loaddata(filename)
        load(filename,'vNewDataset');
end

