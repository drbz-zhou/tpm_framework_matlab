% pick the lowest loss lambda, and redo the NCA, then pick the highest 10,
% 20, 30, 40, 50... features
Cweightmatrix=cell(10,1);
for Person = 1:10
    load(['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA.mat'])
    
    meanloss=mean(lossvals,4);
    meanloss=mean(meanloss,2);
    [minloss,minind]=min(mean(meanloss,1));
    Lambda = lambdavals(minind);
    
    sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
    destfile = ['data/table_v2/part4 result/Table_P',num2str(Person),'_NCA.mat'];

    load(sourcefile);
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
    mdls=cell(17,3);
    for patch = 1:3
        for des = 1:17 %par for here
            disp( ['person ',num2str(Person),' patch ', num2str(patch), ' des ', num2str(des)] )
            X_part = X(:,  (patch-1)*17*39 + (des-1)*39 + 1:(patch-1)*17*39 + (des-1)*39 + 39  );
            if(~isempty(find(isnan(X_part), 1)))
                disp( ['person ',num2str(Person),' patch ', num2str(patch), ' des ', num2str(des),' exists NaN values'] )
            end
            X_part( isnan(X_part) ) = 0;
            mdl = fscnca(X_part,y,'Verbose',0,'Lambda',Lambda);
            mdls{des,patch}=mdl;
        end
    end
    %%
    weightmatrix = zeros(39,17,3);
    for patch = 1:3
        for des = 1:17
            mdl=mdls{des,patch};
            weightmatrix(:,des,patch)=mdl.FeatureWeights;
        end
    end
    FWM=mean(weightmatrix,3); % Feature Weight Matrix
    save(destfile,'mdls','weightmatrix','FWM','Lambda','-v7.3')
    Cweightmatrix{Person,1}=weightmatrix;
end
save('data/table_v2/part4 result/all_weightmatrix.mat','Cweightmatrix','-v7.3');
%% spacial features
% pick the lowest loss lambda, and redo the NCA, then pick the highest 10,
% 20, 30, 40, 50... features
Cweightmatrix=cell(10,1);
num_kf = 8;
num_spacfeats = 10;
for Person = 1:10
    load(['data/table_v2/finetuneNCA/Table_P',num2str(Person),'_NCA_s.mat'])
    
    meanloss=mean(lossvals,4);
    meanloss=mean(meanloss,2);
    [minloss,minind]=min(mean(meanloss,1));
    Lambda = lambdavals(minind);
    
    sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
    destfile = ['data/table_v2/part4 result/Table_P',num2str(Person),'_NCA_s.mat'];

    load(sourcefile);
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
    mdls=cell(num_kf,3);
    for patch = 1:3
        for kf = 1:num_kf %par for here
            disp( ['person ',num2str(Person),' patch ', num2str(patch), ' kf ', num2str(kf)] );
            X_part = X(:,  (patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats+ 1:...
                (patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats + num_spacfeats  );
            if(~isempty(find(isnan(X_part), 1)))
                disp( ['person ',num2str(Person),' patch ', num2str(patch), ' des ', num2str(kf),' exists NaN values'] )
            end
            X_part( isnan(X_part) ) = 0;
            mdl = fscnca(X_part,y,'Verbose',0,'Lambda',Lambda);
            mdls{kf,patch}=mdl;
        end
    end
    %%
    weightmatrix = zeros(num_spacfeats,num_kf,3);
    for patch = 1:3
        for kf = 1:num_kf
            mdl=mdls{kf,patch};
            weightmatrix(:,kf,patch)=mdl.FeatureWeights;
        end
    end
    FWM=mean(weightmatrix,3); % Feature Weight Matrix
    save(destfile,'mdls','weightmatrix','FWM','Lambda','-v7.3')
    Cweightmatrix{Person,1}=weightmatrix;
end
save('data/table_v2/part4 result/all_weightmatrix_s.mat','Cweightmatrix','-v7.3');

%% time domain
FWM_all=zeros(39,17,10);
FWM_mean=zeros(39,17);
FWM_diff=zeros(39,17);
for P=1:10
    FWM_all(:,:,P)=mean(Cweightmatrix{P,1},3);
end
FWM_mean=mean(FWM_all,3);
FWM_diff=mean(abs(FWM_all-repmat(FWM_mean,1,1,10)),3);
%% space domain
FWM_all=zeros(num_spacfeats,num_kf,10);
FWM_mean=zeros(num_spacfeats,num_kf);
FWM_diff=zeros(num_spacfeats,num_kf);
for P=1:10
    FWM_all(:,:,P)=mean(Cweightmatrix{P,1},3);
end
FWM_mean=mean(FWM_all,3);
FWM_diff=mean(abs(FWM_all-repmat(FWM_mean,1,1,10)),3);
%% pick top features
FWM_line=FWM_mean(:);
[B,I]=sort(FWM_line,'descend');
num_feats = 2;
%%
Person = 1;
sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
load(sourcefile);
[C,ia,ic] = unique(all_label);
Ccell=cell(size(C));
for i=1:length(C)
    Cind=find(all_label==C(i));
    CindRand=Cind(randperm(length(Cind)));
    Ccell{i}=CindRand;
end
% don't include class 8 -it's null
numSamples=size(all_feats,2);
for i=1:(length(Ccell)-1)
    numSamples = min([numSamples, length(Ccell{1,i})]);
end
% balance dataset
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

data_part = data(:, [I(1:num_feats); 39*17+I(1:num_feats); 39*17*2+I(1:num_feats); size(data,2)]);
data_part_weighted = [data_part(:,1:end-1).*repmat(B(1:num_feats),3,size(data_part,1))',data_part(:,end)];
%% combine all person's data together, each person is balanced
numSamples=zeros(10,1);
data_all_blc = [];
for Person = 1:10
    sourcefile = ['data/table_v2/Table_P',num2str(Person),'_tempfeats.mat'];
    load(sourcefile);
    all_feats_t=all_feats;
    sourcefile = ['data/table_v2/Table_P',num2str(Person),'_spacfeats.mat'];
    load(sourcefile);
    all_feats_s=all_feats;
    all_feats=[all_feats_t;all_feats_s];
    
    [C,ia,ic] = unique(all_label);
    Ccell=cell(size(C));
    for i=1:length(C)
        Cind=find(all_label==C(i));
        CindRand=Cind(randperm(length(Cind)));
        Ccell{i}=CindRand;
    end
    % don't include class 8 -it's null
    % balance all classes from a person
    numSamples(Person)=size(all_feats,2);
    for i=1:(length(Ccell)-1)
        numSamples(Person) = min([numSamples(Person), length(Ccell{1,i})]);
    end
    newInd=[Ccell{1}(1:numSamples(Person)), Ccell{2}(1:numSamples(Person)),...
        Ccell{3}(1:numSamples(Person)),Ccell{4}(1:numSamples(Person))...
        Ccell{5}(1:numSamples(Person)),Ccell{6}(1:numSamples(Person))...
        Ccell{7}(1:numSamples(Person))];
    X=all_feats(:,newInd)';
    y=all_label(:,newInd)';
    %normalize
    %X=(X-mean(X))./repmat(std(X),size(X,1),1);
    %X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    
    X( isnan(X) ) = 0;
    data=[X,y];
    data_all_blc=[data_all_blc;data];
end
save('data/table_v2/Table_AllP_spacetime_feats.mat','data_all_blc','-v7.3');
%% normalize all together
load('data/table_v2/Table_AllP_spacetime_feats.mat');
X=data_all_blc(:,1:end-1);
X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
y=data_all_blc(:,end);
data_all_n=[X,y];
data_all_blc_t=[data_all_blc(:,1:1989),y];
data_all_blc_s=data_all_blc(:,1990:end);


%% pick top features for all
FWM_line=FWM_mean(:);
[B,I]=sort(FWM_line,'descend');
num_feats = 10;

data_part = data(:, [I(1:num_feats); 39*17+I(1:num_feats); 39*17*2+I(1:num_feats); size(data,2)]);
%data_part = data(:, [I(1:num_feats); num_spacfeats*num_kf+I(1:num_feats);...
%    num_spacfeats*num_kf*2+I(1:num_feats); size(data,2)]);

%data_part_weighted = [data_part(:,1:end-1).*repmat(B(1:num_feats),3,size(data_part,1))',data_part(:,end)];
