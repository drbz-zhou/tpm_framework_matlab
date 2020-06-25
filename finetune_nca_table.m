function finetune_nca_table(sourcefile, destfile, lambdadivs, fold_k, st)
num_des = 17;
num_tempfeats = 39;
num_kf = 8;
num_spacfeats = 10;
%% balance samples
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
    cvp = cvpartition(y,'kfold',fold_k);
    numvalidsets = cvp.NumTestSets;
    lambdavals = linspace(0,lambdadivs,lambdadivs)/length(y);
    if st == 0
        lossvals = zeros(num_des,3,length(lambdavals),numvalidsets);
    elseif st == 1
        lossvals = zeros(num_kf,3,length(lambdavals),numvalidsets);
    end
    
    
    mdls=cell(num_des,3,lambdadivs);
    for i_lambda = 1:length(lambdavals)
        for k = 1:numvalidsets
            Xtrain = X(cvp.training(k),:);
            ytrain = y(cvp.training(k),:);
            Xvalid = X(cvp.test(k),:);
            yvalid = y(cvp.test(k),:);
            for patch = 1:3
                if st == 0
                    for des = 1:num_des %par for here
                        disp(sourcefile)
                        disp( ['i ', num2str(i_lambda), ' k ', num2str(k),...
                            ' patch ', num2str(patch), ' des ', num2str(des)] )
                        X_part = Xtrain(:,  (patch-1)*num_des*num_tempfeats + ...
                            (des-1)*num_tempfeats + 1:(patch-1)*num_des*num_tempfeats + ...
                            (des-1)*num_tempfeats + num_tempfeats  );
                        X_valid_part = Xvalid(:,  (patch-1)*num_des*num_tempfeats +...
                            (des-1)*num_tempfeats + 1:(patch-1)*num_des*num_tempfeats + ...
                            (des-1)*num_tempfeats + num_tempfeats  );

                        if(~isempty(find(isnan(X_part), 1)))
                            disp( ['patch ', num2str(patch), ' des ', num2str(des),' exists NaN values'] )
                        end
                        X_part( isnan(X_part) ) = 0;
                        mdl = fscnca(X_part,ytrain,'Verbose',0, 'Lambda',...
                            lambdavals(i_lambda));
                            %,'IterationLimit',30,...
                            %'GradientTolerance',1e-4, 'Standardize',true);
                        mdls{des,patch,i_lambda}=mdl;
                        lossvals(des,patch,i_lambda,k) = loss(mdl,X_valid_part,yvalid,'LossFunction','classiferror');
                    end
                elseif st == 1
                    for kf = 1:num_kf
                        disp( ['i ', num2str(i_lambda), ' k ', num2str(k),...
                            ' patch ', num2str(patch),' kf ', num2str(kf)] )
                        X_part = Xtrain(:,  (patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats...
                            + 1:(patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats + num_spacfeats  );
                        X_valid_part = Xvalid(:,  (patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats...
                            + 1:(patch-1)*num_kf*num_spacfeats + (kf-1)*num_spacfeats + num_spacfeats  );

                        if(~isempty(find(isnan(X_part), 1)))
                            disp( ['i ', num2str(i_lambda), ' k ', num2str(k),...
                            ' patch ', num2str(patch),' kf ', num2str(kf),' exists NaN values'] )
                        end
                        X_part( isnan(X_part) ) = 0;
                        mdl = fscnca(X_part,ytrain,'Verbose',0, 'Lambda',...
                            lambdavals(i_lambda));
                            %,'IterationLimit',30,...
                            %'GradientTolerance',1e-4, 'Standardize',true);
                        mdls{kf,patch,i_lambda}=mdl;
                        lossvals(kf,patch,i_lambda,k) = loss(mdl,X_valid_part,yvalid,'LossFunction','classiferror');
                    end
                end
            end
        end
    end
    %meanloss = mean(lossvals,2);
    %%
    if st == 0
        weightmatrix = zeros(num_tempfeats,num_des,3);
        for i_lambda =  1:length(lambdavals)
            for patch = 1:3
                for des = 1:num_des
                    mdl=mdls{des,patch,i_lambda};
                    weightmatrix(:,des,patch)=mdl.FeatureWeights;
                end
            end
        end
    elseif st == 1
        weightmatrix = zeros(num_spacfeats,num_kf,3);
        for i_lambda =  1:length(lambdavals)
            for patch = 1:3
                for kf = 1:num_kf
                    mdl=mdls{kf,patch,i_lambda};
                    weightmatrix(:,kf,patch)=mdl.FeatureWeights;
                end
            end
        end
    end
    
    
    save(destfile, 'weightmatrix', 'lossvals','lambdavals','mdls');
    
end