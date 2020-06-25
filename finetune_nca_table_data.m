function finetune_nca_table_data(data, destfile, lambdadivs, fold_k, st)
% data should already be balanced
% this function is used for all people together without the need to balance
% samples
num_des = 17;
num_tempfeats = 39;
num_kf = 8;
num_spacfeats = 10;
X=data(:,1:end-1);
y=data(:,end);
    %%
    cvp = cvpartition(y,'kfold',fold_k);
    numvalidsets = cvp.NumTestSets;
    lambdavals = linspace(0,2,lambdadivs)/length(y);
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