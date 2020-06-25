function sfeats = stream2spacfeats(stream, win_size, win_step, upscale,adcbase)

    % calcualte spacial features
    num_feats = 10;
    num_kf = 8;
    
    len_stream = size(stream,3);
    if win_size < len_stream
        num_win = floor((len_stream-win_size)/win_step);
    else
        num_win = 1;
    end
    
    % upscale
    if upscale > 1
        stream = imresize(stream, upscale);
    end
    % devide with adcbase
    if adcbase > 1
        stream = stream / adcbase;
    end
    
    sfeats = zeros(num_feats*num_kf,num_win);
    for i = 1:num_win
        
        if win_size < len_stream
            temp_stream = stream(:,:, (i-1)*win_step+1 : ((i-1)*win_step+win_size) );    
        else
            temp_stream = stream;
        end
        % calculate key frames
        kf = stream2kf(temp_stream);
        for k=1:num_kf
            sfeats( (k-1)*num_feats + 1 : k*num_feats, i ) = spacfeatures(kf(:,:,k));
        end
    end
end