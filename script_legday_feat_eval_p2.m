% combine all data together and balance, normalize
% the original dataset is already balanced
% normalize together at the end
%Cweightmatrix=cell(10,1);
data_all_blc = [];
for Person = 1:6
%% balance samples
    sourcefile = ['data/LegDay/LegDay_P',num2str(Person),'_tempfeats.mat'];
    load(sourcefile);
    all_feats_t=all_feats;
    sourcefile = ['data/LegDay/LegDay_P',num2str(Person),'_spacfeats.mat'];
    load(sourcefile);
    all_feats_s=all_feats;
    all_feats=[all_feats_t;all_feats_s];
    
    X=all_feats';
    y=all_label'-1;
    %X=(X-mean(X))./repmat(std(X),size(X,1),1);
    %X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    
    X( isnan(X) ) = 0;
    data=[X,y];
    data_all_blc = [data_all_blc; data];
    
end
X=data_all_blc(:,1:end-1);
y=data_all_blc(:,end);
X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
data_all_blc_n=[X,y];
save('data/LegDay/LegDay_All_spacetimefeats.mat','data_all_blc','data_all_blc_n','-v7.3');
%% K fold NCA and Branched NCA
load('data/LegDay/LegDay_All_spacetimefeats.mat');
destfile=('data/LegDay/LegDay_20FoldNCA_Rand.mat');
[weightvector,mdls_k]=KFoldNCA(data_all_blc_n, destfile, 20, 1);
[weightmatrix_t,mdls_t]=branchedNCA(data_all_blc_n(:,[1:663,end]), 'data/LegDay/LegDay_BranchedNCA_Time.mat', 0);
[weightmatrix_s,mdls_s]=branchedNCA(data_all_blc_n(:,664:end), 'data/LegDay/LegDay_BranchedNCA_Space.mat', 1);
%%
load('data\LegDay\LegDay_20FoldNCA_Rand.mat')
%%
load('data\LegDay\LegDay_BranchedNCA_Time.mat')
FWV_t = weightmatrix(:);
load('data\LegDay\LegDay_BranchedNCA_Space.mat')
FWV_s = weightmatrix(:);
FWV = [FWV_t;FWV_s];
%%
[B,I]=sort(FWV,'ascend'); % ascend for least ranks, descend for highest ranks
num_feats = 2;
I_f=I(1:num_feats);
data_part = data_all_blc_n(:, [I_f;end]);
%% plot for paper

load('data/LegDay/LegDay_20FoldNCA_Rand.mat');
plotNCAWeightMatrix(reshape(weightvector(1:663),39,17),0);
plotNCAWeightMatrix(reshape(weightvector(664:end),10,8),1);

