clear; close all; clc;

% Load videos
load('cam1_4.mat');
load('cam2_4.mat');
load('cam3_4.mat');

% Find the number of frames
numFrames1_4 = size(vidFrames1_4,4);
numFrames2_4 = size(vidFrames2_4,4);
numFrames3_4 = size(vidFrames3_4,4);

% Play the video1_3 to create the filter
% implay(vidFrames1_4)

% Difine the x,y width of the filter1_3
x_width_14 = 50;
y_width_14 = 130;
% Create the filter for cam1_4
filter1_4 = zeros(480,640);
filter1_4((300-y_width_14):1:(300+y_width_14), (350-x_width_14):1:(350+x_width_14)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data1_4 = [];
for j = 1:numFrames1_4
    C1_4 = vidFrames1_4(:,:,:,j);
    C_to_Gray_14 = rgb2gray(C1_4);
    Gray1_4 = double(C_to_Gray_14);
    
    Gray1_4f = Gray1_4.*filter1_4;
    white_14 = max(Gray1_4f(:))*0.95;
    thresh1_4 = Gray1_4f > white_14;
    [Y,X] = ind2sub(size(thresh1_4),find(thresh1_4));
    
    data1_4 = [data1_4; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh1_4 * max(Gray1_4f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray1_4f)); drawnow
%     title('Gray1_4f');
end

close all;


% Play the video2_4 to create the filter
% implay(vidFrames2_4)

% Difine the x,y width of the filter2_4
x_width_24 = 50;
y_width_24 = 150;
% Create the filter for cam2_4
filter2_4 = zeros(480,640);
filter2_4((230-y_width_24):1:(230+y_width_24), (300-x_width_24):1:(300+x_width_24)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data2_4 = [];
for j = 1:numFrames2_4
    C2_4 = vidFrames2_4(:,:,:,j);
    C_to_Gray_24 = rgb2gray(C2_4);
    Gray2_4 = double(C_to_Gray_24);
    
    Gray2_4f = Gray2_4.*filter2_4;
    white_24 = max(Gray2_4f(:))*0.95;
    thresh2_4 = Gray2_4f > white_24;
    [Y,X] = ind2sub(size(thresh2_4),find(thresh2_4));
    
    data2_4 = [data2_4; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh2_4 * max(Gray2_4f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray2_4f)); drawnow
%     title('G2_4f');
end

close all;



% Play the video3_4 to create the filter
% implay(vidFrames3_4)

% Difine the x,y width of the filter3_4
x_width_34 = 110;
y_width_34 = 50;
% Create the filter for cam3_4
filter3_4 = zeros(480,640);
filter3_4((290-y_width_34):1:(290+y_width_34), (375-x_width_34):1:(375+x_width_34)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data3_4 = [];
for j = 1:numFrames3_4
    C3_4 = vidFrames3_4(:,:,:,j);
    C_to_Gray_34 = rgb2gray(C3_4);
    Gray3_4 = double(C_to_Gray_34);
    
    Gray3_4f = Gray3_4.*filter3_4;
    white_34 = max(Gray3_4f(:))*0.95;
    thresh3_4 = Gray3_4f > white_34;
    [Y,X] = ind2sub(size(thresh3_4),find(thresh3_4));
    
    data3_4 = [data3_4; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh3_4 * max(Gray3_4f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray3_4f)); drawnow
%     title('G3_4f');
end



%% 
% Find the lowest Y coodinate of each data
[M,I] = min(data1_4(1:50,2));
data1 = data1_4(I:end,:);

[M,I] = min(data2_4(1:50,2));
data2 = data2_4(I:end,:);

[M,I] = min(data3_4(1:50,2));
data3 = data3_4(I:end,:);

% Find the shortest video
min_length = min([length(data1);length(data2);length(data3)]);
% Trimmed other to make them have the same length
data1_new = data1(1:min_length,:);
data2_new = data2(1:min_length,:);
data3_new = data3(1:min_length,:);

% Add all data into a large one
data_all = [data1_new';data2_new';data3_new'];

% Compute mean for each row
mu = mean(data_all,2);
% Subtract mu
data_all_new = data_all - mu;

% Perform SVD
[u,s,v] = svd(data_all_new'/sqrt(min_length-1));
% Generat eigenvalues
lambda = diag(s).^2;

figure()
plot(lambda/sum(lambda),'ro--','Linewidth',2);
title('Test4: Energy of Each Diagonal Variance')
xlabel('Modes')
ylabel('% of Energy Captured')


figure()
X = (1:min_length);
subplot(2,1,1)
plot(X,data_all_new(1,:),X,data_all_new(2,:),'Linewidth',2);
title('Test4: Original displacement (Camera1)')
xlabel('Frames')
ylabel('Displacement (pixels)')
legend('XY plane','Z axis','Location','eastoutside')

% subplot(2,2,2)
% plot(X,data_all_new(3,:),X,data_all_new(4,:),'Linewidth',2);
% title('Case2: Original displacement (Camera2)')
% xlabel('Frames')
% ylabel('Displacement (pixels)')
% legend('XY plane','Z axis')
% 
% 
% subplot(2,2,3)
% plot(X,data_all_new(5,:),X,data_all_new(6,:),'Linewidth',2);
% title('Case2: Original displacement (Camera3)')
% xlabel('Frames')
% ylabel('Displacement (pixels)')
% legend('Z axis','XY plane')


Y = data_all_new' * v;
subplot(2,1,2)
plot(X,Y(:,1),X,Y(:,2),X,Y(:,3),'Linewidth',2);
title('Test4: Displacement across principal component directions')
xlabel('Frames')
ylabel('Displacement (pixels)')
legend('PC1','PC2','PC3','Location','eastoutside')
