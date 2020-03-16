clear; close all; clc;
% Load Testdata
load Testdata

L = 15; % spatial domain
n = 64; % Fourier modes

% Define the domain discretization
x2 = linspace(-L,L,n+1); 
% Consider only the first n points (periodicity)
x = x2(1:n); 
y = x; 
z = x;
% frequency components
k = (2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; 
% Shift the k vector so the frequencies match up (use for plot)
ks = fftshift(k);

% Return 3-D grid coordinates based on the coordinates in x, y, z
[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

% Create variable for average frequency
Utave = zeros(n,n,n);

% Calculate average frequencies for 20 different measurements
for j = 1:20
    % Reshape Undata into a 64-by-64-by-64 array
    Un(:,:,:) = reshape(Undata(j,:),n,n,n);
    Unt(:,:,:) = fftn(Un);
    Utave = Utave + Unt;
end

% -------------------------- Figure 1 ---------------------------------
%
% Draw the isosurface in time domain
figure(1)
isosurface(X,Y,Z,abs(Un),0.4);
axis([-20 20 -20 20 -20 20]), grid on;
xlabel('X');ylabel('Y');zlabel('Z')
set(gca,'Fontsize',12)
% 

Utaves = fftshift(Utave);
absUtaves = abs(Utaves);

% Find the frequency signature (center frequency) in Utave
cfreq = 0;
for ii = 1:n
    for jj = 1:n
        for kk = 1:n
            if absUtaves(ii,jj,kk) > cfreq
                cfreq = absUtaves(ii,jj,kk);
                a = ii; 
                b = jj;
                c = kk;
            end
        end
    end
end

% -------------------------- Figure 2 ---------------------------------

% Create the isosurface of frequency signature
figure(2)
isosurface(Kx,Ky,Kz,absUtaves/cfreq,0.6);
axis([-abs(ks(1)) abs(ks(1)) -abs(ks(1)) abs(ks(1)) -abs(ks(1)) abs(ks(1))]), grid on;
xlabel('Kx');ylabel('Ky');zlabel('Kz');
set(gca,'Fontsize',12);

% Check to see which one of the shift of the k vector belongs to Kx, Ky,
% and Kz
ks(a);
ks(b);
ks(c);

% Create the Gaussian filter with width \tau = 1/2
g = exp(-((Kx-ks(b)).^2 + (Ky - ks(a)).^2 + (Kz-ks(c)).^2)/2);

% Apply the Gaussian filter to denoise the data
Path = [];
for j = 1:20
    Un(:,:,:) = reshape(Undata(j,:),n,n,n);
    Unt(:,:,:) = fftn(Un);
    Unts = fftshift(Unt);
    
    Unts_g = Unts.*g;
    Uns_g = ifftn(Unts_g);
    absUns_g = abs(Uns_g);

    cfreq = 0;
    for ii = 1:n
        for jj = 1:n
            for kk = 1:n
                if absUns_g(ii,jj,kk) > cfreq
                    cfreq = absUns_g(ii,jj,kk);
                    a = X(1,ii,1); b = Y(jj,1,1); c = Z(1,1,kk);
                end
            end
        end
    end
    % Store the indicies of all center frequencies which is the path of the
    % marble
    Path = [Path ; [b a c]];
end

% -------------------------- Figure 3 ---------------------------------
%
% Use plot3 to plot the path of the marble
figure(3)
plot3(Path(:,1),Path(:,2),Path(:,3),'-o','Color','k','LineWidth',2,'MarkerSize',13,'MarkerFaceColor','c');
axis([-20 20 -20 20 -20 20]), grid on;
xlabel('X');ylabel('Y');zlabel('Z');
set(gca,'FontSize',12);

% Print out the location of the marble at the 20th data measurement.
Path(20,:)

% -------------------------- Figure 4 ---------------------------------
%
% Plot the location of the marble at the 20th data measurement.
figure(4)
isosurface(X,Y,Z,absUns_g/cfreq,0.7);
axis([-20 20 -20 20 -20 20]), grid on;
xlabel('X');ylabel('Y');zlabel('Z');
set(gca,'FontSize',12);
