clear; close all; clc

%% 1D plotting - f(x)
figure(1)
x = -10:.01:10;
f = sin(x);
plot(x,f,'Linewidth',2)

%% 2D plotting f(x,y)

% Meshgrid example
x = -1:.5:1;
y = 2:.5:4;
[X,Y] = meshgrid(x,y);

% Now plot in 2D
x = -5:.2:5;
y = -5:.2:5;
[X,Y] = meshgrid(x,y);
f = X.^2+Y.^2;
figure(2)
surf(X,Y,f)

% Contour plot
figure(3)
contour(X,Y,f)

%% 3D plotting - f(x,y,z)
x = -5:.1:5;
y = x;
z = x;
[X,Y,Z] = meshgrid(x,y,z);
f = (X.^2+Y.^2+Z.^2);
figure(4)
isosurface(X,Y,Z,f,1)
axis([-5 5 -5 5 -5 5])

% complex-valued function
f = exp(2i)*(X.^2+Y.^2+Z.^2);
figure(5)
isosurface(X,Y,Z,abs(f),1)
axis([-5 5 -5 5 -5 5])

%% 2D fft example
clear; close all; clc
x = linspace(-pi,pi,80);
y = x;
[X,Y] = meshgrid(x,y);
u = sin(4*X).*sin(2*Y);
surf(X,Y,u)








