% Mary had a little lamb - Piano
clear; close all; clc;

% [y,Fs] = audioread('music1.wav');
[y,Fs] = audioread('music2.wav');


tr_piano = length(y)/Fs;  % record time in seconds
% figure()
% plot((1:length(y))/Fs,y);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Mary had a little lamb (piano)');
% % title('Mary had a little lamb (recorder)');
% p8 = audioplayer(y,Fs); playblocking(p8);

y = y.';
L = tr_piano;
n = length(y);
t = (1:n)/Fs;
k = (2*pi/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);


% % Plot the portion of music in freq domain
% figure()
% yt = fft(y);
% yt_shift = fftshift(yt);
% plot(ks,abs(yt_shift)/max(abs(yt)));
% xlabel('Freq [\omega]');
% ylabel('Amplitude');
% title('Mary had a little lamb (piano) in Freq Domain');
% % title('Mary had a little lamb (recorder) in Freq Domain');

% Use Gabor filter produce spectrograms of the piece of work
a = 40; % Set window width
tslide = 0:0.15:L; % Set time sliding
ygt_spec = [];
for j = 1:length(tslide)
    g = exp(-a*(t-tslide(j)).^2);
    yg = g.*y;  % Apply filter
    ygt = fft(yg);  % Take fft of filtered data
    ygt_spec(j,:) = fftshift(abs(ygt)); % Store fft in spectrogram
    
    % Plot the process of the accomplishment of the Gabor transform
%     subplot(3,1,1), plot(t,v,'k',t,g,'r')
%     xlabel('Time (sec)'), ylabel('Amplitude')
%     title('Gabor Filter and Signal')
%     axis([-50 50 0 1])
%     subplot(3,1,2), plot(t,vg,'k')
%     xlabel('Time (sec)'), ylabel('Amplitude')
%     title('Gabor Filter * Signal')
%     axis([-50 50 0 1])
%     subplot(3,1,3), plot(t,abs(fftshift(vgt))/max(abs(vgt)),'k')
%     xlabel('Time (sec)'), ylabel('Amplitude')
%     title('Gabor Transform of Signal')
%     axis([-50 50 0 1])
%     drawnow
%     pause(0.1)
end
ygt_spec;

% Plot portion of spectrogram
% figure()
% pcolor(tslide,ks/(2*pi),ygt_spec.'),shading interp
% % set(gca,'Ylim',[200 350],'Fontsize',[14])
% set(gca,'Ylim',[700,1100],'Fontsize',[14])
% xlabel('Time (sec)'), ylabel('Frequency (Hz)');
% % title('Piano');
% % title('Record');
% colormap(hot)
% 
% Plot full of spectrogram
figure()
pcolor(tslide,ks/(2*pi),ygt_spec.'),shading interp
set(gca,'Ylim',[200 1500],'Fontsize',[14])
xlabel('Time (sec)'), ylabel('Frequency (Hz)');
% title('Piano')
title('Record')
colormap(hot)

