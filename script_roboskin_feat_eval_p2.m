% the robo skin data is already balanced
% normalize

load('data\RoboSkin\RoboSkin_AllP_spacetime_feats.mat');
num_timefeats=663;
num_spacfeats=80;
X=data_all(:,1:end-1);
X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
y=data_all(:,end);
X_t = X(:,1:num_timefeats);
X_s = X(:,num_timefeats+1:end);
data_all_n=[X,y];
data_all_n_t=[X_t,y];
data_all_n_s=[X_s,y];
save('data\RoboSkin\RoboSkin_AllP_spacetime_feats_n.mat','data_all_n','-v7.3')
%%
load('data\RoboSkin\RoboSkin_AllP_spacetime_feats_n.mat');
%%
[weightmatrix_t,mdls_t]=branchedNCA(data_all_n(:,[1:663,end]), 'data/RoboSkin/RoboSkin_BranchedNCA_Time.mat', 0);
[weightmatrix_s,mdls_s]=branchedNCA(data_all_n(:,664:end), 'data/RoboSkin/RoboSkin_BranchedNCA_Space.mat', 1);
destfile=('data/RoboSkin/RoboSkin_20FoldNCA_Rand.mat');
[weightvector,mdls_k]=KFoldNCA(data_all_n, destfile, 20, 1);
%%
load('E:\OwnCloud\Doctoral\matlab\data\RoboSkin\RoboSkin_20FoldNCA_Rand.mat')
%%
load('E:\OwnCloud\Doctoral\matlab\data\RoboSkin\RoboSkin_BranchedNCA_Time.mat')
FWV_t = weightmatrix(:);
load('E:\OwnCloud\Doctoral\matlab\data\RoboSkin\RoboSkin_BranchedNCA_Space.mat')
FWV_s = weightmatrix(:);
FWV = [FWV_t;FWV_s];
%%
[B,I]=sort(FWV,'ascend'); % ascend for least ranks, descend for highest ranks
num_feats = 2;
I_f=I(1:num_feats);
data_part = data_all_n(:, [I_f;end]);
%% 
load('data/RoboSkin/RoboSkin_20FoldNCA_Rand.mat');
plotNCAWeightMatrix(reshape(weightvector(1:663),39,17),0);
plotNCAWeightMatrix(reshape(weightvector(664:end),10,8),1);
