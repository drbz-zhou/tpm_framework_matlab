% fine tune NCA with separate Des or KF
Cweightmatrix=cell(10,1);
%%
num_tempfeats = 39;
num_kf = 8;
num_spacfeats = 10;
flag_temp = 0;
flag_spac = 1;
for Person = 1%:10
%%
%% balance samples
    if flag_temp == 1
        load(['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat']);
    end
    if flag_spac == 1
        load(['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat']);
    end
    [C,ia,ic] = unique(all_label);
    Ccell=cell(size(C));
    for i=1:length(C)
        Cind=find(all_label==C(i));
        CindRand=Cind(randperm(length(Cind)));
        Ccell{i}=CindRand;
    end
    %% don't include class 8 -it's null
    numSamples=size(all_feats,2);
    for i=1:(length(Ccell)-1)
        numSamples = min([numSamples, length(Ccell{1,i})]);
    end
%%
    newInd=[Ccell{1}(1:numSamples), Ccell{2}(1:numSamples),...
        Ccell{3}(1:numSamples),Ccell{4}(1:numSamples)...
        Ccell{5}(1:numSamples),Ccell{6}(1:numSamples)...
        Ccell{7}(1:numSamples)];
    X=all_feats(:,newInd)';
    y=all_label(:,newInd)';
    %X=(X-mean(X))./repmat(std(X),size(X,1),1);
    X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    
    X( isnan(X) ) = 0;
    data=[X,y];
    %%
    cvp = cvpartition(y,'kfold',5);
    numvalidsets = cvp.NumTestSets;
    lambdadivs = 10;
    lambdavals = linspace(0,lambdadivs,lambdadivs)/length(y);
    
    lossvals = zeros(17,3,length(lambdavals),numvalidsets);
    lossvals_s = zeros(num_kf,3,length(lambdavals),numvalidsets);
    
    
    mdls=cell(17,3,lambdadivs);
    for i_lambda = 1:length(lambdavals)
        for k = 1:numvalidsets
            Xtrain = X(cvp.training(k),:);
            ytrain = y(cvp.training(k),:);
            Xvalid = X(cvp.test(k),:);
            yvalid = y(cvp.test(k),:);
            for patch = 1:3
                %% temp
                if flag_temp == 1
                    for des = 1:17 %par for here
                        disp( ['person ',num2str(Person),' patch ', num2str(patch), ' des ', num2str(des)] )
                        X_part = Xtrain(:,  (patch-1)*17*num_tempfeats + (des-1)*num_tempfeats...
                            + 1:(patch-1)*17*num_tempfeats + (des-1)*num_tempfeats + num_tempfeats  );
                        X_valid_part = Xvalid(:,  (patch-1)*17*num_tempfeats + (des-1)*num_tempfeats...
                            + 1:(patch-1)*17*num_tempfeats + (des-1)*num_tempfeats + num_tempfeats  );
                        
                        if(~isempty(find(isnan(X_part), 1)))
                            disp( ['person ',num2str(Person),' patch ', num2str(patch), ' des ', num2str(des),' exists NaN values'] )
                        end
                        X_part( isnan(X_part) ) = 0;
                        mdl = fscnca(X_part,ytrain,'Verbose',0, 'Lambda',...
                            lambdavals(i_lambda));
                        %,'IterationLimit',30,...
                        %'GradientTolerance',1e-4, 'Standardize',true);
                        mdls{des,patch,i_lambda}=mdl;
                        lossvals(des,patch,i_lambda,k) = loss(mdl,X_valid_part,yvalid,'LossFunction','classiferror');
                    end
                end
                %% spacial
                if flag_spac == 1
                    for kf = 1:num_kf
                        disp( ['person ',num2str(Person),' patch ', num2str(patch), ' kf ', num2str(kf)] )
                        X_part = Xtrain(:,  (patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats...
                            + 1:(patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats + num_spacfeats  );
                        X_valid_part = Xvalid(:,  (patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats...
                            + 1:(patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats + num_spacfeats  );

                        if(~isempty(find(isnan(X_part), 1)))
                            disp( ['person ',num2str(Person),' patch ', num2str(patch), ' kf ', num2str(kf),' exists NaN values'] )
                        end
                        X_part( isnan(X_part) ) = 0;
                        mdl = fscnca(X_part,ytrain,'Verbose',0, 'Lambda',...
                            lambdavals(i_lambda));
                            %,'IterationLimit',30,...
                            %'GradientTolerance',1e-4, 'Standardize',true);
                        mdls{kf,patch,i_lambda}=mdl;
                        lossvals_s(kf,patch,i_lambda,k) = loss(mdl,X_valid_part,yvalid,'LossFunction','classiferror');
                    end
                end
            end
        end
    end
    %meanloss = mean(lossvals,2);
    %%
    if flag_temp == 1
        weightmatrix = zeros(39,17,3);
        for patch = 1:3
            for des = 1:17
                mdl=mdls{des,patch};
                weightmatrix(:,des,patch)=mdl.FeatureWeights;
            end
        end
    end
    if flag_spac == 1
        weightmatrix = zeros(num_spacfeats,num_kf,3);
        for patch = 1:3
            for k = 1:num_kf
                mdl=mdls{k,patch};
                weightmatrix(:,k,patch)=mdl.FeatureWeights;
            end
        end
    end
%%
    Cweightmatrix{Person,1}=weightmatrix;
end

%% the actual part, above are trials for finetune_nca_t
Person = 1;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%
Person = 2;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%
Person = 3;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%
Person = 4;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%
Person = 5;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%%
Person = 6;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%
Person = 7;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%
Person = 8;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%
Person = 9;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%
Person = 10;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 0);
%% spacial
Person = 1;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 2;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 3;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 4;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 5;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 6;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 7;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 8;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 9;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%
Person = 10;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
destfile = ['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'];
finetune_nca_table(sourcefile, destfile, 10, 5, 1);
%% branched NCA with all data
load('data/table_v2/Table_AllP_spacfeats.mat');
Patch1=[data_all_blc_s(:,1:80),data_all_blc_s(:,end)];
Patch2=[data_all_blc_s(:,81:160),data_all_blc_s(:,end)];
Patch3=data_all_blc_s(:,161:end);
FWM_1=branchedNCA(Patch1,[],1);
FWM_2=branchedNCA(Patch2,[],1);
FWM_3=branchedNCA(Patch3,[],1);
FWM_s=(FWM_1+FWM_2+FWM_3);
%%
load('data/table_v2/Table_AllP_tempfeats.mat');
Patch1=[data_all_blc_t(:,1:663),data_all_blc_t(:,end)];
Patch2=[data_all_blc_t(:,664:1326),data_all_blc_t(:,end)];
Patch3=data_all_blc_t(:,1327:end);
FWM_1=branchedNCA(Patch1,[],0);
FWM_2=branchedNCA(Patch2,[],0);
FWM_3=branchedNCA(Patch3,[],0);
FWM_t=(FWM_1+FWM_2+FWM_3);
%% linear space time weight vector
w_st_branch=[FWM_t(:); FWM_s(:)];
w_st_branch_n=[FWM_t(:)/max(FWM_t(:)); FWM_s(:)/max(FWM_s(:))];
%% plot weight matrix
plotNCAWeightMatrix(FWM_t/max(FWM_t(:)),0)
plotNCAWeightMatrix(FWM_s/max(FWM_s(:)),1)
%%
[B,I]=sort(w_st_branch_n,'descend');
II=zeros(size(I));
for i=1:size(I,1)
    II(I(i))=i;
end
%these two are the indexes of the ranking in the matrix form
FWM_t_i=reshape(II(1:663),39,17);
FWM_s_i=reshape(II(664:end),10,8);
plotNCAWeightMatrix(FWM_t_i,0);
plotNCAWeightMatrix(FWM_s_i,1);
%% translate the ranking into which features are at the top
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
%% K fold NCA with all data
load('data/table_v2/Table_AllP_spacetime_feats.mat');
Folds = 20;
Patch1=[data_all_blc(:,[1:663, 1990:2069]),data_all_blc(:,end)];
Patch2=[data_all_blc(:,[664:1326, 2070:2149]),data_all_blc(:,end)];
Patch3=[data_all_blc(:,[1327:1989, 2150:2229]),data_all_blc(:,end)];
%%
FV1=KFoldNCA(Patch1,'data/table_v2/KFoldNCA/Patch1_noRand',Folds,0);
FV2=KFoldNCA(Patch2,'data/table_v2/KFoldNCA/Patch2_noRand',Folds,0);
FV3=KFoldNCA(Patch3,'data/table_v2/KFoldNCA/Patch3_noRand',Folds,0);
%%
FV1=KFoldNCA(Patch1,'data/table_v2/KFoldNCA/Patch1_20_Rand',Folds,1);
FV2=KFoldNCA(Patch2,'data/table_v2/KFoldNCA/Patch2_20_Rand',Folds,1);
FV3=KFoldNCA(Patch3,'data/table_v2/KFoldNCA/Patch3_20_Rand',Folds,1);
%%
FV1=load('data/table_v2/KFoldNCA/Patch1_noRand','weightvector');
FV2=load('data/table_v2/KFoldNCA/Patch2_noRand','weightvector');
FV3=load('data/table_v2/KFoldNCA/Patch3_noRand','weightvector');
%%
FV1r=load('data/table_v2/KFoldNCA/Patch1_20_Rand','weightvector');
FV2r=load('data/table_v2/KFoldNCA/Patch2_20_Rand','weightvector');
FV3r=load('data/table_v2/KFoldNCA/Patch3_20_Rand','weightvector');
FVr = FV1r.weightvector + FV2r.weightvector + FV3r.weightvector;
FVr = FVr/max(FVr);
%%
FV = FV1.weightvector+FV2.weightvector+FV3.weightvector;
FV = FV / max(FV);
%%
subplot(1,3,1)
imagesc(weightmatrix(:,:,1))
subplot(1,3,2)
imagesc(weightmatrix(:,:,2))
subplot(1,3,3)
imagesc(weightmatrix(:,:,3))
%% visualize mean loss
meanloss=mean(lossvals,4);
meanloss=mean(meanloss,2);
meanloss = permute(meanloss, [1,3,2]);
figure('color','white');
plot(meanloss')
figure('color','white');
plot(mean(meanloss,1))
%%
%sumweightmatrixall = zeros(39,17);
sumweightmatrixall = zeros(num_spacfeats,num_kf);
for Person = 1:10
end
for Person = 1:10
    weightmatrix=Cweightmatrix{Person,1};
    meanweight=mean(weightmatrix,3);
    sumweightmatrixall=sumweightmatrixall+meanweight;
end
meanweightall = sumweightmatrixall / 10;
%diffweight = zeros(39,17);
diffweight = zeros(num_spacfeats,num_kf);
for Person = 1:10
    weightmatrix=Cweightmatrix{Person,1};
    meanweight=mean(weightmatrix,3);
    diffweight = diffweight+abs(meanweightall-meanweight);
end
diffweight=diffweight/10;
plotWeightMatrix(meanweightall);
%caxis([0 5]);
plotWeightMatrix(diffweight);
%caxis([0 5]);
%%
function plotWeightMatrix(weightmatrix)
    figure('Color','white')
    imagesc(weightmatrix)
    colormap('copper')
    colorbar
    for x = 1:size(weightmatrix,1)
        for y=1:size(weightmatrix,2)
            text(y,x,num2str(weightmatrix(x,y),'%.2f'),'Color',[0.9,0.0,0.5],'HorizontalAlignment','center','FontSize',12);
        end
    end

    set(gca,'YTick', 1:size(weightmatrix,2), 'FontSize', 12)
    set(gca,'XTick', 1:size(weightmatrix,1), 'FontSize', 12)

    ax = gca;
    ax.XLabel.String = 'Frame Descriptor';
    ax.XLabel.FontSize = 12;
    ax.YLabel.String = 'Temporal Feature';
    ax.YLabel.FontSize = 12;
end