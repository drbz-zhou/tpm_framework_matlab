function tfeat=tempfeatures(Des)
    % Des should be a 1-by-T temporal sequence
    Des = Des(:);
    numFeat = 39;
    tfeat = zeros(numFeat,1);
    % tfeat 1 average
    tfeat(1) = mean(Des);
    % tfeat 2 variance
    tfeat(2) = var(Des);
    % tfeat 3 range
    tfeat(3) = max(Des)-min(Des);
    % tfeat 4 skewness
    tfeat(4) = skewness(Des);
    % tfeat 5 kurtosis
    tfeat(5) = kurtosis(Des);
    % tfeat 6 waveform length (ref year 2012)
    tfeat(6) = sum(diff(Des));
    % tfeat 7 WAMP with threshold as mean(Des)
    tfeat(7) = sum(Des(Des>mean(Des)));
    % maybe add MAD here? mean absolute deviation
    
    % spectrum features
    % calculate power spectrum density
    PSD = calspectrum(Des);
    % tfeat 8 mean magnitude
    tfeat(8) = mean(PSD);
    % tfeat 9 mean frequency
    x = 1:length(PSD);
    tfeat(9) = sum(PSD.*x(:)) / sum(PSD);
    % tfeat 10, 11, 12, 13, 14  five bands of spectrum
    xband = floor(length(PSD)/5);
    tfeat(10) = mean(PSD(2 : xband));
    tfeat(11) = mean(PSD(xband+1 : xband*2));
    tfeat(12) = mean(PSD(xband*2+1 : xband*3));
    tfeat(13) = mean(PSD(xband*3+1 : xband*4));
    tfeat(14) = mean(PSD(xband*4+1 : end));
    
    % wavelet features
    tfeat(15:39) = calwavelet(Des);
end

function P1 = calspectrum(S)
    % single sided spectrum
    L = length(S);
    Y = fft(S);
    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
end

function WaveletFeat = calwavelet(Des)
    J = 4;
    [c,info] = fwt(Des,'db8',J);
    CC = wavpack2cell(c,info.Lc,info.dim);
    numJFeat = 5;
    WaveletFeat = zeros( (J+1)*numJFeat,1);
    for j=1:length(CC)
        tempC=CC{j};
        if length(tempC) < 10  
            % if array too short, cannot calculate skewness and kurtosis
            tempC = padarray(tempC, ceil((10-length(tempC))/2));
        end
        WaveletFeat((j-1)*numJFeat + 1) = mean(tempC);
        WaveletFeat((j-1)*numJFeat + 2) = var(tempC);
        WaveletFeat((j-1)*numJFeat + 3) = max(tempC)-min(tempC);
        WaveletFeat((j-1)*numJFeat + 4) = skewness(tempC);
        WaveletFeat((j-1)*numJFeat + 5) = kurtosis(tempC);
    end
end