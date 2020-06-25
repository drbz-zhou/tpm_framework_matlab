data=MatrixData;
plot(mean(LinearData,2));
%%
for i=1209%1:size(data,3)
    f=data(:,:,i);
    imagesc(f)
    pause(0.05)
end
%%
F=data(:,:,1209)';
fig1=figure('color', 'white');
imagesc(F);
colormap(gray);
AX=gca;
set(AX, 'Units', 'pixels', 'Position', [50, 50, 400, 400]);

fig2=figure(2);
HistF = histogram(F(:));

fig3=figure('color', 'white');
BF = F>90;
imagesc(BF)
colormap(gray);
AX=gca;
set(AX, 'Units', 'pixels', 'Position', [50, 50, 400, 400]);
%%
min(F(BF))
max(F(BF))
%%
fig4=figure('color', 'white');
imagesc(imresize(F,4,'bicubic'));
colormap(gray);
AX=gca;
set(AX, 'Units', 'pixels', 'Position', [50, 50, 400, 400]);
%%
R=zeros(256*3,1);
G=zeros(256*3,1);
B=zeros(256*3,1);
R(1:256)=1:256; R(257:768)=256;
G(257:512)=1:256; G(513:768)=256;
B(513:768)=1:256;
figure('color', 'white');
plot(R,'DisplayName','R','color','red');hold on;
plot(G,'DisplayName','G','color','green');
plot(B,'DisplayName','B','color','blue');
I=mean([R,G,B],2);
plot(I,'color','black')
hold off;
