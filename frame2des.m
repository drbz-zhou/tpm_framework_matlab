function Des=frame2des(inF)
    %inF should be normalized according to its absolute value ranges
    % e.g. for 12 bit, inF should be the raw value devided by 4096
    numDes = 17;
    Des = zeros(numDes,1);
    % Des 1 mean
    Des(1) = mean(inF(:));
    % Des 2 variance
    Des(2) = var(inF(:));
    % Des 3 range
    Des(3) = max(inF(:))-min(inF(:));
    % Des 4 entropy
    Des(4) = entropy(inF);
    % Des 5 mean absolute deviation
    Des(5) = mad(inF(:));
    
    % Des 6 & 7 center of mass
    [Des(6), Des(7)] = frame2CoM(inF);
    
    % threshold to get binary
    TF=mean(inF(:))-(mean(inF(:))-min(inF(:)))/4;
    BinF = double(inF>TF);
    
    % Des 8 & 9 centroid of thresholded 
    [Des(8),Des(9)] = frame2CoM(BinF);
    
    % Des 10 area
    Des(10) = sum(BinF(:))/length(BinF(:));
    
    % Hu's 7 moments
    
    moments = frame2moments(inF);
    Des(11:17)=moments(4:10);
    
end

function [x, y]=frame2CoM(inF)
    
    [M, N] = size(inF);
    [x, y] = meshgrid(1:N, 1:M);
    x = x(:);
    y = y(:);
    F = inF(:);
    x =  mean(x .* F)/mean(F);
    y =  mean(y .* F)/mean(F);
    x = x/M;
    y = y/N;
    %x =  abs(x - N/2);
    %y =  abs(y - M/2);
end