function boutonDetection(img)
sc = 8;
img = mat2gray(img);
th = graythresh(img);
img(img<th) = 0;
im(img,2);
win = round(8*sc);
logf = fspecial('log',[win win],sc);
logf = -logf;
logf = logf/sum(abs(logf));
A = imfilter(img,logf,'replicate','same');
A(A<0) = 0;
Amax = ordfilt2(A,25,ones(5,5));
[r,c] = find(Amax==A & img > 0.2);
%     A = imfilter(imfilter(img,logf,'replicate','same'),logf','replicate','same');
% im(A); 
hold on;
plot(c,r,'r.');

