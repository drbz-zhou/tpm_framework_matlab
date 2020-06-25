function kf = stream2kf(inS)
    num_kf = 8;
    kf=zeros(size(inS,1),size(inS,2),num_kf);
    % kf 1 mean frame
    kf(:,:,1)=mean(inS,3);
    
    % kf 2 mean diff
    dfS=diff(inS,1,3);
    kf(:,:,2)=sum(dfS,3);
    kf(:,:,3)=sum(dfS(dfS>0));
    kf(:,:,4)=sum(abs(dfS(dfS<0)));
    
    
    meanS = mean(mean(inS,1),2);
    meanS = meanS(:);
    [~,I]=max(meanS);
    kf(:,:,5) = inS(:,:,I);
    [~,I]=min(meanS);
    kf(:,:,6) = inS(:,:,I);
    
    stdS = std(std(inS,0,1),0,2);
    stdS = stdS(:);
    [~,I]=max(stdS);
    kf(:,:,7) = inS(:,:,I);
    
    A=zeros(size(inS,1),size(inS,2));
    for i=1:size(inS,3)
        temp_f = inS(:,:,i);
        A=A+reshape(temp_f.*(temp_f>meanS(i)), size(A));
    end
    kf(:,:,8)=A/size(inS,3);
end