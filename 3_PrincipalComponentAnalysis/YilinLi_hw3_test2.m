clear; close all; clc;

% Load videos
load('cam1_2.mat');
load('cam2_2.mat');
load('cam3_2.mat');

% Find the number of frames
numFrames1_2 = size(vidFrames1_2,4);
numFrames2_2 = size(vidFrames2_2,4);
numFrames3_2 = size(vidFrames3_2,4);

% Play the video1_2 to create the filter
% implay(vidFrames1_2)

% Difine the x,y width of the filter1_2
x_width_12 = 60;
y_width_12 = 110;
% Create the filter for cam1_2
filter1_2 = zeros(480,640);
filter1_2((310-y_width_12):1:(310+y_width_12), (370-x_width_12):1:(370+x_width_12)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data1_2 = [];
for j = 1:numFrames1_2
    C1_2 = vidFrames1_2(:,:,:,j);
    C_to_Gray_12 = rgb2gray(C1_2);
    Gray1_2 = double(C_to_Gray_12);
    
    Gray1_2f = Gray1_2.*filter1_2;
    white_12 = max(Gray1_2f(:))*0.95;
    thresh1_2 = Gray1_2f > white_12;
    [Y,X] = ind2sub(size(thresh1_2),find(thresh1_2));
    
    data1_2 = [data1_2; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh1_2 * max(Gray1_2f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray1_2f)); drawnow
%     title('Gray1_2f');
end

close all;


% Play the video2_2 to create the filter
% implay(vidFrames2_2)

% Difine the x,y width of the filter2_2
x_width_22 = 105;
y_width_22 = 170;
% Create the filter for cam2_2
filter2_2 = zeros(480,640);
filter2_2((250-y_width_22):1:(250+y_width_22), (315-x_width_22):1:(315+x_width_22)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data2_2 = [];
for j = 1:numFrames2_2
    C2_2 = vidFrames2_2(:,:,:,j);
    C_to_Gray_22 = rgb2gray(C2_2);
    Gray2_2 = double(C_to_Gray_22);
    
    Gray2_2f = Gray2_2.*filter2_2;
    white_22 = max(Gray2_2f(:))*0.95;
    thresh2_2 = Gray2_2f > white_22;
    [Y,X] = ind2sub(size(thresh2_2),find(thresh2_2));
    
    data2_2 = [data2_2; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh2_2 * max(Gray2_2f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray2_2f)); drawnow
%     title('G2_2f');
end

close all;



% Play the video3_2 to create the filter
% implay(vidFrames3_2)

% Difine the x,y width of the filter3_2
x_width_32 = 100;
y_width_32 = 70;
% Create the filter for cam3_2
filter3_2 = zeros(480,640);
filter3_2((270-y_width_32):1:(270+y_width_32), (400-x_width_32):1:(400+x_width_32)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data3_2 = [];
for j = 1:numFrames3_2
    C3_2 = vidFrames3_2(:,:,:,j);
    C_to_Gray_32 = rgb2gray(C3_2);
    Gray3_2 = double(C_to_Gray_32);
    
    Gray3_2f = Gray3_2.*filter3_2;
    white_32 = max(Gray3_2f(:))*0.95;
    thresh3_2 = Gray3_2f > white_32;
    [Y,X] = ind2sub(size(thresh3_2),find(thresh3_2));
    
    data3_2 = [data3_2; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh3_2 * max(Gray3_2f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray3_2f)); drawnow
%     title('G3_2f');
end



%% 
% Find the lowest Y coodinate of each data
[M,I] = min(data1_2(1:50,2));
data1 = data1_2(I:end,:);

[M,I] = min(data2_2(1:50,2));
data2 = data2_2(I:end,:);

[M,I] = min(data3_2(1:50,2));
data3 = data3_2(I:end,:);

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
title('Test2: Energy of Each Diagonal Variance')
xlabel('Modes')
ylabel('% of Energy Captured')


figure()
X = (1:min_length);
subplot(2,1,1)
plot(X,data_all_new(1,:),X,data_all_new(2,:),'Linewidth',2);
title('Test2: Original displacement (Camera1)')
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
plot(X,Y(:,1),X,Y(:,2),X,Y(:,3),X,Y(:,4),X,Y(:,5),'Linewidth',2);
title('Test2: Displacement across principal component directions')
xlabel('Frames')
ylabel('Displacement (pixels)')
legend('PC1','PC2','PC3','PC4','PC5','Location','eastoutside')

