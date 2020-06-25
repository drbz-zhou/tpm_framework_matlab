Cweightmatrix=cell(10,1);
for Person = 1:10
%% balance samples
    load(['data/Table_P',num2str(Person),'_tempfeats.mat']);
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
        parfor des = 1:17 %par for here
            disp( ['person ',num2str(Person),' patch ', num2str(patch), ' des ', num2str(des)] )
            X_part = X(:,  (patch-1)*17*39 + (des-1)*39 + 1:(patch-1)*17*39 + (des-1)*39 + 39  );
            if(~isempty(find(isnan(X_part), 1)))
                disp( ['person ',num2str(Person),' patch ', num2str(patch), ' des ', num2str(des),' exists NaN values'] )
            end
            X_part( isnan(X_part) ) = 0;
            mdl = fscnca(X_part,y,'Verbose',0);
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
    Cweightmatrix{Person,1}=weightmatrix;
end

%%
subplot(1,3,1)
imagesc(weightmatrix(:,:,1))
subplot(1,3,2)
imagesc(weightmatrix(:,:,2))
subplot(1,3,3)
imagesc(weightmatrix(:,:,3))
%%
sumweightmatrixall = zeros(39,17);
for Person = 1:10
end
for Person = 1:10
    weightmatrix=Cweightmatrix{Person,1};
    meanweight=mean(weightmatrix,3);
    sumweightmatrixall=sumweightmatrixall+meanweight;
end
meanweightall = sumweightmatrixall / 10;
diffweight = zeros(39,17);
for Person = 1:10
    weightmatrix=Cweightmatrix{Person,1};
    meanweight=mean(weightmatrix,3);
    diffweight = diffweight+abs(meanweightall-meanweight);
end
diffweight=diffweight/10;
plotWeightMatrix(meanweightall);
caxis([0 5]);
plotWeightMatrix(diffweight);
caxis([0 5]);

