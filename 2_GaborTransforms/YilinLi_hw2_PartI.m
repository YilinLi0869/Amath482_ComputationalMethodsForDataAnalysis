% Handel's Messiah
clear; close all; clc;

load handel
v = y';

% % Plot the portion of music I will analyze. 
% figure(1)
% plot((1:length(v))/Fs,v);
% xlabel('Time [sec]');
% ylabel('Amplitude');
% title('Signal of Interest, v(n)');

% % To play this back in MATLAB:
% p8 = audioplayer(v,Fs);
% playblocking(p8);

L = (length(v)-1) / Fs; % Length of the piece
% Identify first and last points of v
v = v(1:end-1); % periodic
n = length(v);
t = (1:n)/ Fs;
k = (2*pi/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

% % Plot the portion of music in freq domain
% figure(2)
% vt = fft(v);
% vt_shift = fftshift(vt);
% plot(ks,abs(vt_shift)/max(abs(vt)));
% xlabel('Freq [\omega]');
% ylabel('Amplitude');
% title('Signal of Interest in Freq Domain');

% Use Gabor filter produce spectrograms of the piece of work
% Set window width
a = 1; 
% a = 0.2;
% a = 20;
% a = 50;

% Different translations
dt = 0.1;
% dt = 0.01;
% dt = 0.5;
% dt = 1;
tslide = 0:dt:L; % Set time sliding
vgt_spec = [];
for j = 1:length(tslide)
    % % Gaussian Window
    % g = exp(-a*(t-tslide(j)).^2);
    % % Mexican Hat Wavelet
    % g = (1-((t-tslide(j))/a).^2).*exp(-(((t-tslide(j))/a).^2)/2);
    % % Step-function (Shannon) Window
    g = (abs(t-tslide(j)) < a/2);
    vg = g.*v;  % Apply filter
    vgt = fft(vg);  % Take fft of filtered data
    vgt_spec(j,:) = fftshift(abs(vgt)); % Store fft in spectrogram
    
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
vgt_spec;
figure()
pcolor(tslide,ks,vgt_spec.'),shading interp
colormap(hot)
xlabel('Time [sec]'),ylabel('Frequency [Hz]')





