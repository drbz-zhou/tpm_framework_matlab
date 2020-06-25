figure('color','white');
for i=1:5
    subplot(2,5,i)
    plotCM(D(:,:,i)/3300) 
    subplot(2,5,5+i)
    plotCM(A(:,:,i)/3300) 
end
%%
RowMin = min(D,[],2);
RowMinMat = repmat(RowMin, 1,5,1);
A=D-RowMinMat;