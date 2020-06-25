function tfeats = stream2tempfeats(stream, win_size, win_step, win_mode, upscale,adcbase)
% convert stream to temporal sequence of Des
% the 3rd dimension is time T
numDes = 17;
numFeat = 39;
len_stream = size(stream,3);

Des=zeros(17, len_stream);
% upscale
if upscale > 1
    stream = imresize(stream, upscale);
end
% devide with adcbase
if adcbase > 1
    stream = stream / adcbase;
end
% convert to frame descriptors
for i = 1:size(stream,3)
    frame=stream(:,:,i);
    Des(:,i) = frame2des(frame);
end
% sliding window
%win_size = 100;
%win_step = win_size/4;
if win_size < len_stream
    num_win = floor((len_stream-win_size)/win_step);
else
    num_win = 1;
end
%win_mode = 0;
% window function
% win_mode = 1 - hanning window, win_mode = 0 - rectangular window
win_fun = tukeywin(win_size, win_mode);


tfeats = zeros(numDes*numFeat,num_win);

for D = 1:numDes
    for i = 1:num_win
        if win_size < len_stream
            temp_Des = Des(D, (i-1)*win_step+1 : ((i-1)*win_step+win_size) ).*win_fun;
            tfeats((D-1)*numFeat+1:D*numFeat, i) = tempfeatures(temp_Des);
        else
            tfeats(((D-1)*numFeat+1):(D*numFeat), i) = tempfeatures(Des(D,:).*win_fun);
        end
    end
end
end