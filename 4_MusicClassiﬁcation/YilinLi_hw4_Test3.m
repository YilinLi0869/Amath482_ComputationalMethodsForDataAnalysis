clear; close all; clc;

% Homework 4
% Test 3: Genre Classification
str = ["https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Yakov_Golman/Piano__orchestra_1/Yakov_Golman_-_07_-_Rainbow.mp3","https://files.freemusicarchive.org/storage-freemusicarchive-org/music/Music_for_Video/Lloyd_Rodgers/Cartesian_Reunion_Memorial_Orchestra_the_little_prince-a_ballet_in_two_acts/Lloyd_Rodgers_-_08_-_On_Questions_of_Responsibility_Act_II.mp3","https://files.freemusicarchive.org/storage-freemusicarchive-org/music/none_given/Scott_Holmes/Scott_Holmes_-_Singles/Scott_Holmes_-_Driven_To_Success.mp3","https://files.freemusicarchive.org/storage-freemusicarchive-org/music/Ziklibrenbib/The_Inventors/Counting_backwards/The_Inventors_-_04_-_Melon.mp3","https://files.freemusicarchive.org/storage-freemusicarchive-org/music/Ziklibrenbib/Mela/Mela_two/Mela_-_03_-_Horrible.mp3","https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Dee_Yan-Key/facts_of_life/Dee_Yan-Key_-_08_-_Blue.mp3"];

A = [];

for jj = 1:length(str)
    
    [y,Fs] = webread(str(jj));
    Fs = Fs/2;
    y = y(1:2:end,:);
    song = [];
    for j = 1:length(y)
        song(j,1) = (y(j,1) + y(j,2))/2;
    end
    song = song(find(song,1,'first'):find(song,1,'last'));
    
    
    for k = 1:5:125
        test = song(Fs*k : Fs*(k+5),1);
        vector = abs(spectrogram(test));
        vector = reshape(vector,[length(vector)*8,1]);
        A = [A vector];
    end
    
end

train1 = A(:, 1:50);
train2 = A(:, 51:100);
train3 = A(:, 101:150);

[U,S,V,threshold1,threshold2,w,sorttype1,sorttype2,sorttype3] = music_trainer(train1,train2,train3);

str2 = ["https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Yakov_Golman/Piano__orchestra_1/Yakov_Golman_-_08_-_Valse_orchestral.mp3","https://files.freemusicarchive.org/storage-freemusicarchive-org/music/none_given/Scott_Holmes/Scott_Holmes_-_Singles/Scott_Holmes_-_-_Teamwork.mp3","https://files.freemusicarchive.org/storage-freemusicarchive-org/music/Ziklibrenbib/Mela/Mela_two/Mela_-_07_-_The_Darker_Side.mp3"];

B = [];

for zz = 1:length(str2)
    
    [y2,Fs2] = webread(str2(zz));
    Fs2 = Fs2/2;
    y2 = y2(1:2:end,:);
    song2 = [];
    for j = 1:length(y2)
        song2(j,1) = (y2(j,1) + y2(j,2))/2;
    end
    song2 = song2(find(song2,1,'first'):find(song2,1,'last'));
    
    
    for k = 1:5:125
        test2 = song2(Fs2*k : Fs2*(k+5),1);
        vector2 = abs(spectrogram(test2));
        vector2 = reshape(vector2,[length(vector2)*8,1]);
        B = [B vector2];
    end
    
end

test1 = B(:,1:25);
test2 = B(:,26:50);
test3 = B(:,51:75);

Test = [test1 test2 test3];
TestMat = U' * Test;  % PCA projection
pval = w' * TestMat;  % LDA projection

% Rock = 0, Jazz = 1, Classical = 2
ResVec = [];
for kk = 1:length(pval)
    if pval(kk) <= threshold1
        ResVec(kk) = 0;
    elseif threshold1 < pval(kk) <= threshold2
        ResVec(kk) = 1;
    else
        ResVec(kk) = 2;
    end
end
ResVec = ResVec';
hiddenlabels = [2*ones(25,1);zeros(25,1);ones(25,1)];
disp('Rate of success');
sucRate = 0;
for jj = 1:75
    if ResVec(jj) == hiddenlabels(jj)
        sucRate = sucRate + 1;
    end
end

sucRate = sucRate / 75

function [U,S,V,threshold1,threshold2,w,sorttype1,sorttype2,sorttype3] = music_trainer(train1_0,train2_0,train3_0)
n1 = size(train1_0,2);
n2 = size(train2_0,2);
n3 = size(train3_0,2);
A = [train1_0 train2_0 train3_0];
[U,S,V] = svd(A,'econ');
songs = S*V';
U = U(:,1:20);
type1 = songs(1:20, 1:n1);
type2 = songs(1:20, (n1+1):(n1+n2));
type3 = songs(1:20, (n1+n2+1):(n1+n2+n3));

m1 = mean(type1, 2);
m2 = mean(type2, 2);
m3 = mean(type3, 2);
m = (m1+m2+m3)/3;

Sw = 0; % within class variances
for k=1:n1
    Sw = Sw + (type1(:,k)-m1)*(type1(:,k)-m1)';
end
for k=1:n2
    Sw = Sw + (type2(:,k)-m2)*(type2(:,k)-m2)';
end
for k=1:n3
    Sw = Sw + (type3(:,k)-m3)*(type3(:,k)-m3)';
end
    
Sb = ((m1-m)*(m1-m)'+(m2-m)*(m2-m)'+(m3-m)*(m3-m)')/3; % between class 
    
[V2,D] = eig(Sb,Sw); % linear discriminant analysis
[~,ind] = max(abs(diag(D)));
w = V2(:,ind); w = w/norm(w,2);
    
vtype1 = w'*type1; 
vtype2 = w'*type2;
vtype3 = w'*type3;

mean(vtype1) % max
mean(vtype2) % min
mean(vtype3) % middle

sorttype1 = sort(vtype1);
sorttype2 = sort(vtype2);
sorttype3 = sort(vtype3);

t1 = length(sorttype2);
t2 = 1;
while sorttype2(t1)>sorttype3(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold1 = (sorttype2(t1)+sorttype3(t2))/2;

t3 = length(sorttype3);
t4 = 1;
while sorttype3(t3)>sorttype1(t4)
    t3 = t3 - 1;
    t4 = t4 + 1;
end
threshold2 = (sorttype3(t3)+sorttype1(t4))/2;

subplot(3,1,1)
histogram(sorttype1,50);hold on, plot([threshold1 threshold1], [0 7],'r')
hold on, plot([threshold2 threshold2], [0 7],'g');
set(gca,'Xlim',[-1000 5000],'Ylim',[0 7])
title('classical')
subplot(3,1,2)
histogram(sorttype2,50);hold on, plot([threshold1 threshold1], [0 7],'r')
hold on, plot([threshold2 threshold2], [0 7],'g');
set(gca,'Xlim',[-1000 5000],'Ylim',[0 7])
title('rock')
subplot(3,1,3)
histogram(sorttype3,50);hold on, plot([threshold1 threshold1], [0 7],'r')
hold on, plot([threshold2 threshold2], [0 7],'g');
set(gca,'Xlim',[-1000 5000],'Ylim',[0 7])
title('jazz')

end
