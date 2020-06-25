% convert stream to temporal sequence of Des
numDes = 17;
numFeat = 44;
len_stream = size(stream,3);

Des=zeros(17, len_stream);
for i = 1:size(stream,3)
    Des(:,i) = frame2des(stream(:,:,i));
end
% sliding window
win_size = 100;
win_step = win_size/4;
num_win = floor((len_stream-win_size)/win_step);
win_mode = 0;
% window function
switch win_mode
    case 0 % rectangular window
        win_fun = ones(win_size,1);
    case 1 % hamming window
        win_fun = hamming(win_size);
    case 2 % tukey window
        win_fun = tukeywin(win_size, 0.2);
end

tfeats = zeros(numDes*numFeat,num_win);
labels = zeros(1, num_win);

for D = 1:numDes
    for i = 1:num_win
        temp_Des = Des(D, (i-1)*win_step+1 : ((i-1)*win_step+win_size) ).*win_fun;
        tfeats((D-1)*numFeat+1:D*numFeat, i) = tempfeatures(temp_Des);
    end
end