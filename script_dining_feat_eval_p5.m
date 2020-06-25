% this script does the features selection NCA once for all the features,
% this will use SGD and takes significantly long time
%% time features
load('data\table_v2\Table_AllP_tempfeats.mat');
mdl = fscnca(data_all_blc_t(:,1:end-1), data_all_blc_t(:,end),'Verbose',1);
%
weightmatrix = reshape(mdl.FeatureWeights,39,17,3);
FWM_mean=mean(weightmatrix,3);
plotNCAWeightMatrix(FWM_mean,0);
save('data\table_v2\AllTogetherNCA\Time once NCA.mat','data_all','mdl','FWM_mean','-v7.3');
% space features
load('data\table_v2\Table_AllP_spacfeats.mat');
mdl = fscnca(data_all_blc_s(:,1:end-1), data_all_blc_s(:,end),'Verbose',1);
%
weightmatrix = reshape(mdl.FeatureWeights,10,8,3);
FWM_mean=mean(weightmatrix,3);
plotNCAWeightMatrix(FWM_mean,1);
save('data\table_v2\AllTogetherNCA\Space once NCA.mat','data_all','mdl','FWM_mean','-v7.3');
% space time features
load('data\table_v2\Table_AllP_spacetime_feats.mat');
mdl = fscnca(data_all_blc(:,1:end-1), data_all_blc(:,end),'Verbose',1);
save('data\table_v2\AllTogetherNCA\Space Time once NCA.mat','data_all','mdl','-v7.3');
%% lambda sweep
clearvars
sourcefile='data\table_v2\Table_AllP_spacetime_feats.mat';
destfile = 'data\table_v2\AllTogetherNCA\L_sweep_spacetime.mat';
lambdadivs = 10;
fold_k=5;

load(sourcefile);
disp(['loaded file: ', sourcefile]);
finetune_nca(data_all_blc(:,1:end-1), data_all_blc(:, end), destfile, lambdadivs, fold_k);

%
clearvars
sourcefile='data\table_v2\Table_AllP_spacfeats.mat';
destfile = 'data\table_v2\AllTogetherNCA\L_sweep_space.mat';
lambdadivs = 10;
fold_k=5;

load(sourcefile);
disp(['loaded file: ', sourcefile]);
finetune_nca(data_all_blc_s(:,1:end-1), data_all_blc_s(:, end), destfile, lambdadivs, fold_k);

%
clearvars
sourcefile='data\table_v2\Table_AllP_tempfeats.mat';
destfile = 'data\table_v2\AllTogetherNCA\L_sweep_time.mat';
lambdadivs = 10;
fold_k=5;

load(sourcefile);
disp(['loaded file: ', sourcefile]);
finetune_nca(data_all_blc_t(:,1:end-1), data_all_blc_t(:, end), destfile, lambdadivs, fold_k);

%% use minimum lambda
clearvars
sourcefile = 'data\table_v2\AllTogetherNCA\L_sweep_spacetime.mat'; % for the loss and lambda
load(sourcefile);
sourcefile = 'data\table_v2\Table_AllP_spacetime_feats.mat'; % for the data
destfile = 'data\table_v2\AllTogetherNCA\minloss_Lambda_spacetime.mat';
load(sourcefile);
meanloss = mean(lossvals,2);
[~,I]=min(meanloss);
mdl = fscnca(data_all_blc(:,1:end-1), data_all_blc(:,end),'Verbose',1,'Lambda',lambdavals(I));
save(destfile,'mdl','-v7.3');

%
clearvars
sourcefile = 'data\table_v2\AllTogetherNCA\L_sweep_space.mat';
load(sourcefile);
sourcefile='data\table_v2\Table_AllP_spacfeats.mat';
destfile = 'data\table_v2\AllTogetherNCA\minloss_Lambda_space.mat';
load(sourcefile);
meanloss = mean(lossvals,2);
[~,I]=min(meanloss);
mdl = fscnca(data_all_blc_s(:,1:end-1), data_all_blc_s(:,end),'Verbose',1,'Lambda',lambdavals(I));
save(destfile,'mdl','-v7.3');

%
clearvars
sourcefile = 'data\table_v2\AllTogetherNCA\L_sweep_time.mat';
load(sourcefile);
sourcefile='data\table_v2\Table_AllP_tempfeats.mat';
destfile = 'data\table_v2\AllTogetherNCA\minloss_Lambda_time.mat';
load(sourcefile);
meanloss = mean(lossvals,2);
[~,I]=min(meanloss);
mdl = fscnca(data_all_blc_t(:,1:end-1), data_all_blc_t(:,end),'Verbose',1,'Lambda',lambdavals(I));
save(destfile,'mdl','-v7.3');
%% plot the feature weights
load('data/table_v2/AllTogetherNCA/minloss_Lambda_spacetime.mat')
w_st=mdl.FeatureWeights;
load('data/table_v2/AllTogetherNCA/minloss_Lambda_time.mat')
w_t =mdl.FeatureWeights;
load('data/table_v2/AllTogetherNCA/minloss_Lambda_space.mat')
w_s =mdl.FeatureWeights;
%
w_t_sum=sum(reshape(w_t,663,3),2);
w_t_sum=w_t_sum/max(w_t_sum);

w_s_sum=sum(reshape(w_s,80,3),2);
w_s_sum=w_s_sum/max(w_s_sum);
w_s_t=[w_t_sum;w_s_sum];
w_s_t=w_s_t/max(w_s_t);


w_st=[sum(reshape(w_st(1:1989),663,3),2);sum(reshape(w_st(1990:end),80,3),2)];
w_st=w_st/max(w_st);
%% ! plot for paper
FV1r=load('data/table_v2/KFoldNCA/Patch1_20_Rand','weightvector');
FV2r=load('data/table_v2/KFoldNCA/Patch2_20_Rand','weightvector');
FV3r=load('data/table_v2/KFoldNCA/Patch3_20_Rand','weightvector');
FVr = FV1r.weightvector + FV2r.weightvector + FV3r.weightvector;
FVr = FVr/max(FVr);
%%
plotNCAWeightMatrix(reshape(FVr(1:663),39,17),0);
plotNCAWeightMatrix(reshape(FVr(664:end),10,8),1);
%%
load('data/table_v2/branched NCA FWM.mat');
w_st_branch=[FWM_t(:);FWM_s(:)];
%%
figure('Color','white');
subplot(4,1,1)
hold on
set(gca,'YGrid', 'on');
plot(w_st,'LineStyle','none','Marker','o','MarkerSize',6,'MarkerEdgeColor','r')
bar(w_st,1,'FaceColor','r','EdgeColor','none')
line([663.5,663.5],[0,1],'Color',[0 0 0],'LineWidth',2);
set(gca,'fontname','times','FontSize',12);

subplot(4,1,2)
hold on
set(gca,'YGrid', 'on');
plot(w_s_t,'LineStyle','none','Marker','*','MarkerSize',6,'MarkerEdgeColor','b')
bar(w_s_t,1,'FaceColor','b','EdgeColor','none')
line([663.5,663.5],[0,1],'Color',[0 0 0],'LineWidth',2);
ylabel('Normalized Feature Weight')
set(gca,'fontname','times','FontSize',12);

subplot(4,1,3)
hold on
set(gca,'YGrid', 'on');
plot(w_st_branch/max(w_st_branch),'LineStyle','none','Marker','<','MarkerSize',6,'MarkerEdgeColor',[1 0.5 0])
bar(w_st_branch/max(w_st_branch),1,'FaceColor',[1 0.5 0],'EdgeColor','none')
set(gca,'fontname','times','FontSize',12);
line([663.5,663.5],[0,1],'Color',[0 0 0],'LineWidth',2);

subplot(4,1,4)
hold on
set(gca,'YGrid', 'on');
plot(FVr,'LineStyle','none','Marker','>','MarkerSize',6,'MarkerEdgeColor',[0 0.5 0.5])
bar(FVr,1,'FaceColor',[0 0.5 0.5],'EdgeColor','none')
xlabel('Feature Index')
set(gca,'fontname','times','FontSize',12);
line([663.5,663.5],[0,1],'Color',[0 0 0],'LineWidth',2);
%legend('all-in-one NCA','domain split NCA','branched NCA','time-space domain split')
%%
plotNCAWeightMatrix(reshape(w_t_sum,39,17),0)
plotNCAWeightMatrix(reshape(w_s_sum,10,8)',1)
%%
plotNCAWeightMatrix(reshape(w_st(1:663),39,17),0)
plotNCAWeightMatrix(reshape(w_st(664:end),10,8)',1)
%%
load('data/table_v2/Table_AllP_spacetime_feats.mat')
X=data_all_blc(:,1:end-1);
X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
y=data_all_blc(:,end);
data_all_n=[X,y];
X_3 = X(:, [1:663,1990:2069])+ X(:, [664:1326,2070:2149])+ X(:, [1327:1989, 2150:2229]);
X_3=(X_3-repmat(min(X_3),size(X_3,1),1))./repmat(max(X_3)-min(X_3),size(X_3,1),1);
data_3_n =[X_3,y];
%% select best/worst weighted features
% this is only for table cloth because it has 3 patches
[B,I]=sort(FVr,'descend'); % ascend for least ranks, descend for highest ranks
num_feats = 2;
I_f=I(1:num_feats);
I_t=I_f(I_f<664);
I_s=I_f(I_f>663)-663;
data_part = data_all_n(:, [I_t; 39*17+I_t; 39*17*2+I_t;...
                    1989+I_s;1989+80+I_s;1989+160+I_s; size(data_all_n,2)]);
%% USE PCA as coefficient as weight vector
load('data/table_v2/KNNPCA.mat');
%%
FV_PCA = abs(0.5-KNNPCA.PCACenters)'./mean(abs(KNNPCA.PCACoefficients),2);
%bar(FV_PCA)
%%
FV_PCA=reshape(FV_PCA,743,3);
FV_PCA=mean(FV_PCA,2);
%% add random data
X_random = rand(size(X));
data_random=[X,X_random,y];
%%
Folds=60;
KFoldNCA(data_random,'data/table_v2/KFoldNCA/addedRandom',Folds,1);
%% translate the ranking into which features are at the top
[B,I]=sort(FVr,'descend'); % ascend for least ranks, descend for highest ranks
rankresult=zeros(size(I,1),3); % first dim: 0/1 for time/space, then feature, then des/KF
for i=1:size(I,1)
    if I(i)<664
        rankresult(i,1)=0;
        rankresult(i,2)=mod(I(i),39);
        if rankresult(i,2)==0
            rankresult(i,2)=39;
        end
        rankresult(i,3)=ceil(I(i)/39);
    else
        rankresult(i,1)=1;
        rankresult(i,2)=mod(I(i)-663,10);
        if rankresult(i,2)==0
            rankresult(i,2)=10;
        end
        rankresult(i,3)=ceil((I(i)-663)/10);
    end
end

%% find the common top ranking features
FC=[];
num_top=160;
for i=1:num_top
    f=rankresult_1(i,:);
    for j=1:num_top
        if f == rankresult_2(j,:)
            for k=1:num_top
                if f == rankresult_3(k,:)
                    FC=[FC; f];
                end
            end
        end
    end
end