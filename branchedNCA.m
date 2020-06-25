function [weightmatrix,mdls]=branchedNCA(data, destfile, st)
% datas shouldn't have any NaN
num_des = 17;
num_tempfeats = 39;
num_kf = 8;
num_spacfeats = 10;
X=data(:,1:end-1);
X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    
y=data(:,end);
%%
if st==0
    mdls=cell(num_des);
    weightmatrix=zeros(num_tempfeats,num_des);
    for des = 1:num_des %par for here
        disp( [' des ', num2str(des)] )
        X_part = X(:,  (des-1)*num_tempfeats + 1: ...
            (des-1)*num_tempfeats + num_tempfeats  );
        
        mdl = fscnca(X_part,y,'Verbose',0);
        mdls{des}=mdl;
        weightmatrix(:,des)=mdl.FeatureWeights;
    end
elseif st==1
    mdls=cell(num_kf);
    weightmatrix=zeros(num_spacfeats,num_kf);
    for kf = 1:num_kf
        disp( [' kf ', num2str(kf)] )
        X_part = X(:, (kf-1)*num_spacfeats + 1: ...
            (kf-1)*num_spacfeats + num_spacfeats  );
        
        mdl = fscnca(X_part,y,'Verbose',0);
        mdls{kf}=mdl;
        weightmatrix(:,kf)=mdl.FeatureWeights;
    end
end
if length(destfile)>0
    save(destfile, 'weightmatrix', 'mdls');
end

end