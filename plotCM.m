function fh=plotCM(cm,f_name,cap_pre)
%% modified for phd thesis sim plot
numClasses=size(cm,1);
colorHigh = [255, 255, 255];
colorLow = [50, 50, 50];
% colorHigh = [255, 255, 57];
% colorLow = [10, 100, 255];
% cmap=zeros(64,3);
% for i=1:64
%     cmap(i,:)=colorHigh-(colorHigh-colorLow)*(i-1)/64;
% end
% cmap=cmap/255;
cmap=zeros(256,3);
for i=1:256
    cmap(i,:)=colorHigh-(colorHigh-colorLow)*(i-1)/256;
end
cmap=cmap/255;
cmap(end,:) = [0,0,0];
if exist('f_name','var')
    fh=figure('Name', f_name,'color','white');
else
    %fh=figure('color','white');
end
if ~exist('cap_pre','var')
    cap_pre='';
end
m_p = cm./repmat(sum(cm,1), size(cm,1),1);
%m_p(isnan(cm)) = 0;
m_r = cm./repmat(sum(cm,2) , 1, size(cm,1));
m_r(isnan(cm)) = 0;
m_f = 2*m_p.*m_r./(m_p+m_r);
m_f(isnan(m_f)) = 0;

% imagesc(cm);
imagesc(cm, [0,1]);
colormap(cmap);
ylabels = {
    %     'stir(M)',...
    %     'scoop(M)',...
    %     'cut(M)',...
    %     'poke(M)',...
    %     'scoop(S)',...
    %     'poke(S)',...
    %     'water',...
    %     'none',...
    '1',...
    '2',...
    '3',...
    '4',...
    '5',...
    '6',...
    '7',...
    '8',...
    '9',...
    '10',...
    '11',...
    '12',...
    '13'
    };
%'look back right',...
%'look back left',...

set(gca,'YTick', 1:numClasses, 'YtickLabel', '', 'FontSize', 10)
set(gca,'XTick', 1:numClasses, 'XtickLabel', '', 'FontSize', 20)
%set(gcf,'OuterPosition',[0,0,500,500])
% xlabel('Ground Truth','FontSize',20);
% ylabel('Predictons','FontSize',20);
% 

accuracy = sum(diag(m_r))/sum(sum(m_r));

precision = diag(cm)./sum(cm,1)';
precision(isnan(cm)) = 0;
recall = diag(cm)./sum(cm,2);
recall(isnan(cm)) = 0;
fscore = 2 * precision.*recall./(precision+recall);
fscore(isnan(fscore)) = 0;
%mean_F1=mean(fscore);
mean_F1=mean(diag(m_f));

for k = 1:numClasses
    for l=1:numClasses
        if cm(l,k)~=0
            text(k,l,num2str(cm(l,k),'%.2f'),'Color',[0.4,0.4,0.1],'HorizontalAlignment','center','FontSize',12);
            
        end
    end
end
AccStr = num2str(round(accuracy*1000)/1000);
title([cap_pre,'F1: ', num2str(round(mean_F1*1000)/1000), ' ACC: ', AccStr],'FontSize',20);%, 'precision:', num2str(mean(precision)), ' recall:', num2str(mean(recall))));