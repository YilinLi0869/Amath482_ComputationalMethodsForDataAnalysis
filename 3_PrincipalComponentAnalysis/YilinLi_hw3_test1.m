clear; close all; clc;

% Load videos
load('cam1_1.mat');
load('cam2_1.mat');
load('cam3_1.mat');

% Find the number of frames
numFrames1_1 = size(vidFrames1_1,4);
numFrames2_1 = size(vidFrames2_1,4);
numFrames3_1 = size(vidFrames3_1,4);

% Play the video1_1 to create the filter
% implay(vidFrames1_1)

% Difine the x,y width of the filter1_1
x_width_11 = 50;
y_width_11 = 130;
% Create the filter for cam1_1
filter1_1 = zeros(480,640);
filter1_1((300-y_width_11):1:(300+y_width_11), (350-x_width_11):1:(350+x_width_11)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data1_1 = [];
for j = 1:numFrames1_1
    C1_1 = vidFrames1_1(:,:,:,j);
    C_to_Gray_11 = rgb2gray(C1_1);
    Gray1_1 = double(C_to_Gray_11);
    
    Gray1_1f = Gray1_1.*filter1_1;
    white_11 = max(Gray1_1f(:))*0.95;
    thresh1_1 = Gray1_1f > white_11;
    [Y,X] = ind2sub(size(thresh1_1),find(thresh1_1));
    
    data1_1 = [data1_1; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh1_1 * max(Gray1_1f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray1_1f)); drawnow
%     title('Gray1_1f');
end

close all;


% Play the video2_1 to create the filter
% implay(vidFrames2_1)

% Difine the x,y width of the filter2_1
x_width_21 = 50;
y_width_21 = 150;
% Create the filter for cam2_1
filter2_1 = zeros(480,640);
filter2_1((230-y_width_21):1:(230+y_width_21), (300-x_width_21):1:(300+x_width_21)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data2_1 = [];
for j = 1:numFrames2_1
    C2_1 = vidFrames2_1(:,:,:,j);
    C_to_Gray_21 = rgb2gray(C2_1);
    Gray2_1 = double(C_to_Gray_21);
    
    Gray2_1f = Gray2_1.*filter2_1;
    white_21 = max(Gray2_1f(:))*0.95;
    thresh2_1 = Gray2_1f > white_21;
    [Y,X] = ind2sub(size(thresh2_1),find(thresh2_1));
    
    data2_1 = [data2_1; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh2_1 * max(Gray2_1f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray2_1f)); drawnow
%     title('G2_1f');
end

close all;



% Play the video3_1 to create the filter
% implay(vidFrames3_1)

% Difine the x,y width of the filter3_1
x_width_31 = 110;
y_width_31 = 50;
% Create the filter for cam3_1
filter3_1 = zeros(480,640);
filter3_1((290-y_width_31):1:(290+y_width_31), (375-x_width_31):1:(375+x_width_31)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data3_1 = [];
for j = 1:numFrames3_1
    C3_1 = vidFrames3_1(:,:,:,j);
    C_to_Gray_31 = rgb2gray(C3_1);
    Gray3_1 = double(C_to_Gray_31);
    
    Gray3_1f = Gray3_1.*filter3_1;
    white_31 = max(Gray3_1f(:))*0.95;
    thresh3_1 = Gray3_1f > white_31;
    [Y,X] = ind2sub(size(thresh3_1),find(thresh3_1));
    
    data3_1 = [data3_1; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh3_1 * max(Gray3_1f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray3_1f)); drawnow
%     title('G3_1f');
end



%% 
% Find the lowest Y coodinate of each data
[M,I] = min(data1_1(1:50,2));
data1 = data1_1(I:end,:);

[M,I] = min(data2_1(1:50,2));
data2 = data2_1(I:end,:);

[M,I] = min(data3_1(1:50,2));
data3 = data3_1(I:end,:);

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
title('Test1: Energy of Each Diagonal Variance')
xlabel('Modes')
ylabel('% of Energy Captured')

figure()
X = (1:min_length);
subplot(2,1,1)
plot(X,data_all_new(1,:),X,data_all_new(2,:),'Linewidth',2);
title('Test1: Original displacement (Camera1)')
xlabel('Frames')
ylabel('Displacement (pixels)')
legend('XY plane','Z axis','Location','eastoutside')

% subplot(2,2,2)
% plot(X,data_all_new(3,:),X,data_all_new(4,:),'Linewidth',2);
% title('Case1: Original displacement (Camera2)')
% xlabel('Frames')
% ylabel('Displacement (pixels)')
% legend('XY plane','Z axis')
% 
% 
% subplot(2,2,3)
% plot(X,data_all_new(5,:),X,data_all_new(6,:),'Linewidth',2);
% title('Case1: Original displacement (Camera3)')
% xlabel('Frames')
% ylabel('Displacement (pixels)')
% legend('Z axis','XY plane')


Y = data_all_new' * v;
subplot(2,1,2)
plot(X,Y(:,1),X,Y(:,2),X,Y(:,3),'Linewidth',2);
title('Test1: Displacement across principal component directions')
xlabel('Frames')
ylabel('Displacement (pixels)')
legend('PC1','PC2','PC3','Location','eastoutside')
