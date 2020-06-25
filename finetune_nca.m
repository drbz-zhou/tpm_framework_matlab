function finetune_nca(X, y, destfile, lambdadivs, fold_k)

% data should already be balanced before loading this function! the other
% NCA files that reads a file containing all_feats and all_label may not be
% balanced
    %X=(X-mean(X))./repmat(std(X),size(X,1),1);
    X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    
    X( isnan(X) ) = 0;
    %%
    cvp = cvpartition(y,'kfold',fold_k);
    numvalidsets = cvp.NumTestSets;
    lambdavals = linspace(0,2,lambdadivs)/length(y);
    lossvals = zeros(length(lambdavals),numvalidsets);
    
    
    mdls=cell(lambdadivs);
    for i_lambda = 1:length(lambdavals)
        for k = 1:numvalidsets
            Xtrain = X(cvp.training(k),:);
            ytrain = y(cvp.training(k),:);
            Xvalid = X(cvp.test(k),:);
            yvalid = y(cvp.test(k),:);
            
            
            mdl = fscnca(Xtrain,ytrain,'Verbose',0, 'Lambda',...
                lambdavals(i_lambda));
            
            mdls{i_lambda}=mdl;
            lossval_now = loss(mdl,Xvalid,yvalid,'LossFunction','classiferror');
            lossvals(i_lambda,k) = lossval_now;
            disp(['Lambda: ', num2str(i_lambda), ' fold: ', num2str(k), ' loss: ', num2str(lossval_now)]);
        end
    end
    %meanloss = mean(lossvals,2);

    save(destfile, 'lossvals','lambdavals','mdls');
    
end