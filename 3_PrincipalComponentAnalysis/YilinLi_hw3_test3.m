clear; close all; clc;

% Load videos
load('cam1_3.mat');
load('cam2_3.mat');
load('cam3_3.mat');

% Find the number of frames
numFrames1_3 = size(vidFrames1_3,4);
numFrames2_3 = size(vidFrames2_3,4);
numFrames3_3 = size(vidFrames3_3,4);

% Play the video1_3 to create the filter
% implay(vidFrames1_3)

% Difine the x,y width of the filter1_3
x_width_13 = 60;
y_width_13 = 105;
% Create the filter for cam1_3
filter1_3 = zeros(480,640);
filter1_3((315-y_width_13):1:(315+y_width_13), (340-x_width_13):1:(340+x_width_13)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data1_3 = [];
for j = 1:numFrames1_3
    C1_3 = vidFrames1_3(:,:,:,j);
    C_to_Gray_13 = rgb2gray(C1_3);
    Gray1_3 = double(C_to_Gray_13);
    
    Gray1_3f = Gray1_3.*filter1_3;
    white_13 = max(Gray1_3f(:))*0.95;
    thresh1_3 = Gray1_3f > white_13;
    [Y,X] = ind2sub(size(thresh1_3),find(thresh1_3));
    
    data1_3 = [data1_3; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh1_3 * max(Gray1_3f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray1_3f)); drawnow
%     title('Gray1_3f');
end

close all;


% Play the video2_3 to create the filter
% implay(vidFrames2_3)

% Difine the x,y width of the filter2_3
x_width_23 = 115;
y_width_23 = 120;
% Create the filter for cam2_3
filter2_3 = zeros(480,640);
filter2_3((290-y_width_23):1:(290+y_width_23), (305-x_width_23):1:(305+x_width_23)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data2_3 = [];
for j = 1:numFrames2_3
    C2_3 = vidFrames2_3(:,:,:,j);
    C_to_Gray_23 = rgb2gray(C2_3);
    Gray2_3 = double(C_to_Gray_23);
    
    Gray2_3f = Gray2_3.*filter2_3;
    white_23 = max(Gray2_3f(:))*0.95;
    thresh2_3 = Gray2_3f > white_23;
    [Y,X] = ind2sub(size(thresh2_3),find(thresh2_3));
    
    data2_3 = [data2_3; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh2_3 * max(Gray2_3f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray2_3f)); drawnow
%     title('G2_3f');
end

close all;



% Play the video3_3 to create the filter
% implay(vidFrames3_3)

% Difine the x,y width of the filter3_3
x_width_33 = 100;
y_width_33 = 135;
% Create the filter for cam3_3
filter3_3 = zeros(480,640);
filter3_3((205-y_width_33):1:(205+y_width_33), (370-x_width_33):1:(370+x_width_33)) = 1;

% Convert videos to grayscale
% Use filter to create a window for tracking movement
% Find the point that has the max intensity
% Save x and y coordinates of that point
data3_3 = [];
for j = 1:numFrames3_3
    C3_3 = vidFrames3_3(:,:,:,j);
    C_to_Gray_33 = rgb2gray(C3_3);
    Gray3_3 = double(C_to_Gray_33);
    
    Gray3_3f = Gray3_3.*filter3_3;
    white_33 = max(Gray3_3f(:))*0.95;
    thresh3_3 = Gray3_3f > white_33;
    [Y,X] = ind2sub(size(thresh3_3),find(thresh3_3));
    
    data3_3 = [data3_3; mean(X),mean(Y)];
    
    % Plot to check
%     subplot(1,2,1)
%     imshow(uint8((thresh3_3 * max(Gray3_3f(:))))); drawnow
%     title('Thresh');
%     subplot(1,2,2)
%     imshow(uint8(Gray3_3f)); drawnow
%     title('G3_3f');
end



%% 
% Find the lowest Y coodinate of each data
[M,I] = min(data1_3(1:50,2));
data1 = data1_3(I:end,:);

[M,I] = min(data2_3(1:50,2));
data2 = data2_3(I:end,:);

[M,I] = min(data3_3(1:50,2));
data3 = data3_3(I:end,:);

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
title('Test3: Energy of Each Diagonal Variance')
xlabel('Modes')
ylabel('% of Energy Captured')

figure()
X = (1:min_length);
subplot(2,1,1)
plot(X,data_all_new(1,:),X,data_all_new(2,:),'Linewidth',2);
title('Test3: Original displacement (Camera1)')
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
plot(X,Y(:,1),X,Y(:,2),X,Y(:,3),X,Y(:,4),'Linewidth',2);
title('Test3: Displacement across principal component directions')
xlabel('Frames')
ylabel('Displacement (pixels)')
legend('PC1','PC2','PC3','PC4','Location','eastoutside')