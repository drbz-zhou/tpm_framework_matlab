function [weightvector,mdls]=KFoldNCA(data, destfile, Folds, Randomize)
% data shouldn't have any NaN
num_dim = size(data,2)-1;
num_n = floor(num_dim/Folds);

X=data(:,1:end-1);
X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);

y=data(:,end);
%%
mdls=cell(Folds);
weightvector=zeros(num_dim,1);
randinds=randperm(num_dim);
for k = 1:Folds %par for here
    disp( [' fold ', num2str(k)] )
    if Randomize == 0
        if k<Folds
            inds = (k-1)*num_n + 1 : k*num_n;
        else
            inds = (k-1)*num_n + 1 : num_dim;
        end
    else
        if k<Folds
            inds = randinds((k-1)*num_n + 1 : k*num_n);
        else
            inds = randinds((k-1)*num_n + 1 : num_dim);
        end
    end
    X_part = X(:,inds);
    X_part(isnan(X_part))=0;
    X_part(isinf(X_part))=0;
    mdl = fscnca(X_part,y,'Verbose',0);
    mdls{k}=mdl;
    weightvector(inds)=mdl.FeatureWeights;
end


if length(destfile)>0
    save(destfile, 'weightvector', 'mdls');
end

end