function [img1,bk] = removeBg(img)
mthresh = 2*1e-2;
sthresh = 0.2;
sthresh = sthresh*sthresh;

fprintf('Median filtering\n');
img = medfilt2(img,[3,3]);
img = mat2gray(img);
sz = size(img);
I = zeros(sz);
img1 = zeros(sz);
I2 = zeros(sz);

fprintf('integrate along x\n');
for i=1:sz(1)
    n = i;
    I(n) = img(n);
    for j=2:sz(2)
        n = n+sz(1);
        I(n) = I(n-sz(1)) + img(n);
        I2(n) = I2(n-sz(1)) + img(n)*img(n);
    end
end
fprintf('integrate along y\n');
for j=1:sz(2)
    n = 1+(j-1)*sz(1); %first col coord
    for i=2:sz(1)
        n = n+1;
        I(n) = I(n-1) + I(n);
        I2(n) = I2(n-1) + I2(n);
    end
end
h = 15;
bk = zeros(sz);
fprintf('mainloop ')
for i=1:sz(1)
    for j=1:sz(2)
       x1 = max(1,j-h); 
       y1 = max(1,i-h); 
       x2 = min(sz(2),j+h); 
       y2 = min(sz(1),i+h); 
       a = y1 + sz(1)*(x1-1);
       b = y2 + sz(1)*(x1-1);
       c = y1 + sz(1)*(x2-1);
       d = y2 + sz(1)*(x2-1);
       num = ((x2-x1+1)*(y2-y1+1));
%        num = (2*h+1)^2;
       mn = ( I(d) - I(b) - I(c) + I(a) ) / num;
       mn2 = ( I2(d) - I2(b) - I2(c) + I2(a) ) / num;
       s = max(sthresh,(mn2 - mn*mn));
       if  mn > mthresh
           r = sqrt(1 / s);
           %        img1(i,j) = s;
           img1(i,j) = max(r*(img(i,j) - mn),0);
           bk(i,j) = mn;
       end
    end
    if mod(i,300)==0, fprintf('.'); end
end
fprintf('done!! \n');
img1 = max(0,min(img1,1));

