
clear all;
close all;

% Define filename and path here
dir = '';
dataName1 = 'data\hand.txt';
exportName = 'data\hand.mat';

%%
disp('Loading files');
fid1 = fopen(dataName1);
if(fid1 == -1)
    disp(strcat('fopen ', dir, dataName1, 'failed'));
    return;
end

%%
disp('Scanning data');
clear C1 C2 C3 R1 R2 R3;

formatCell='%3c'; %1 node, 3 byte per node
size_x = 32;
size_y = 32;
FM = formatCell;
FM_length = size_x*size_y;

for i=1:FM_length-1
    FM=strcat(FM,formatCell);
end
%%
C1 = textscan(fid1, strcat('%2s%2s%2s%3s%1s', FM, '%s%s','%*[^\n]'),'delimiter',':', 'endofline','\n');%, 'emptyvalue',0);
%%
T = str2num(char(C1{1}))*(60*60*1000)+str2num(char(C1{2}))*(60*1000)+str2num(char(C1{3}))*(1000)+str2num(char(C1{4}));
DT=diff(T);
R1=zeros(size(C1{1},1),size_x*size_y);%prelocate 400 = 20*20
%%
for i=1:size_x*size_y
    index = i+5; % 4 timestamp numbers and ","
    temp = hex2dec(char(C1{index}));
    R1(1:size(temp,1),i) = temp; 
end
%%
length_min = size(R1,1);
time_ms = T(1:length_min);
%%
%reshape them to 20x20
visualize = false;

stream_cropped = zeros(size_x, size_y,size(R1,1)); %prelocate
for i=1:size(R1,1)

    frame = reshape(R1(i,1:size_x*size_y),size_x,size_y);
    stream_cropped(:,:,i)=frame;
    if visualize == true
        % visualize, comment if running automatically
        imagesc(frame);
        pause(0.05);
    end
end
%%
disp('Saving Data');
%data = permute(stream_cropped, [2 3 1]);
LinearData = R1;
MatrixData = stream_cropped;
save(strcat(dir,exportName),'LinearData','MatrixData','time_ms','-v7.3');
disp ('Free memory');
clear all;

