function preproImg = preprocessing(u0,s, tilesize)
%because the image size is big, we want to do a simple vertical tile based
%processing. The procesing is local, therefore global parameters are not
%needed.
preproImg = [];
if nargin < 1
    fprintf('Input argument missing image file (of class uint8)');
    return;
end

if ~exist('tilesize','var')
    tilesize = 200;
end
if ~exist('s','var') 
    s = 2.5;
end

sz = size(u0);
numtiles = ceil(sz(2)/tilesize);
%readjust tilesize 
tilesize = round(sz(2)/numtiles);
overlap = 20;
preproImg = zeros(sz);

for tile = 1:numtiles
    fprintf('Processing tile %d of %d\n',tile,numtiles);
    start = tilesize*(tile-1)+1;
    start = max(1,start);
    npad = min(start-1, overlap);
    n0pad = overlap - npad;
    
    endt = tilesize*(tile);
    endt = min(sz(2),endt);
    ppad = min(sz(2)-endt, overlap);
    p0pad = overlap - ppad;
    tiledata = cat(2, zeros(sz(1),n0pad, class(u0)),  ...
        u0(1:sz(1), (start-npad) : endt+ppad), ...
        zeros(sz(1),p0pad, class(u0) ));
%    ppout = processTile2(tiledata,s);
    ppout = processTile(tiledata,s);
    x1 = n0pad + npad + 1;
    x2 = size(ppout,2) - (ppad+p0pad);
    preproImg(:,start:endt) = ppout(:,x1:x2);
end

function V = processTile2(u0, s)
%this function computes the vesselness measure (multiscale)
u0 = mat2gray(u0);
alpha = 0.5;
c = 0.25;
ss = (1:0.6:s+2);
V = zeros(size(u0));
for sss = ss
    s1 = round(4*sss);
    g = fspecial('log',s1,sss);
    g = -1*g./sum(abs(g(:)));
    v = imfilter(u0,g,'same','replicate');
%     [~, ~, ~, uxx, uxy, uyy] = getHessianAndST(u0,sss);
%    im([uxx, uxy, uyy]); pause(3);
    %lam2 is higher lam1 is large negetaive
%     [lam2, lam1] = getEigVec2Dsym(uxx, uxy, uyy);
%     a = abs(lam1)<abs(lam2);
%     alam1 = a.*lam1 + (1-a).*lam2; 
%     alam2 = (1-a).*lam1 + a.*lam2;
%     
%     b = abs(alam2) < 1e-8;
%     alam2(b) = sign(alam2(b))*1e-8;
%     
%     Rb = alam1./alam2;
%     S = sqrt(lam1.^2 + lam2.^2);
%     eB = Rb.^2 / (2*alpha*alpha);
%     eC = S.^2 / (2*c*c);
%     v = (exp(-1*eB)).*(1 - exp(-1*eC));
%     b = alam2<0;
%     v(b) = 0;
    V = max(V,v);
end

function W = processTile(u0,s)
%this function computes the Weingarten Matrix and uses the largest eigen
%vector, s is the smoothing parameter
%W = F2 * inv(F1)
%where
%   F2 = 1/(1+Ix^2+Iy^2) + [Ixx Ixy; Iyx Iyy]
%   F1 = [1+Ix2 IxIy; IxIy 1+Iy2]

if ~strcmp(class(u0),'uint8')
    u0 = mat2gray(u0);
end

[ux2, uxuy, uy2, uxx, uxy, uyy] = getHessianAndST(u0,s);
F1inv = getF1Inverse(1+ux2, uxuy, 1+uy2);

den = sqrt(1+ux2+uy2);
F2(:,:,1) = -1*uxx./den;
F2(:,:,2) = -1*uxy./den;
F2(:,:,3) = -1*uyy./den;

W1 = F1inv(:,:,1).*F2(:,:,1) + F1inv(:,:,2).*F2(:,:,2);
W2 = F1inv(:,:,1).*F2(:,:,2) + F1inv(:,:,2).*F2(:,:,3);
W3 = F1inv(:,:,2).*F2(:,:,2) + F1inv(:,:,3).*F2(:,:,3);

lambda = getEigVec2Dsym(W1, W2, W3);
W = abs(lambda);
W = threshold(W);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function W = threshold(W)
[gx,gy] = gradient(W);
gn = sqrt(gx.^2 + gy.^2);
gn  = mat2gray(gn);
n = find(gn > 0.01);
thresh = 0.5* sum(W(n).*gn(n))/sum(gn(n));
W(W<thresh) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function F1inv = getF1Inverse(p, q, r)
F1inv = zeros(size(p,1), size(p,2), 3);
det = p.*r - q.^2;
F1inv(:,:,1) = r./det;
F1inv(:,:,2) = -q./det;
F1inv(:,:,3) = p./det;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ux2, uxuy, uy2, uxx, uxy, uyy] = getHessianAndST(u0,s)
%cumputes the derivatives using FFT
u = gauss(u0, s);
[ny, nx] = size(u0);
%first derv
ux = 0.5*(u(:,[2:nx nx])-u(:,[1 1:nx-1]));
uy = 0.5*(u([2:ny ny],:)-u([1 1:ny-1],:));

%second derv
uxx = u(:,[2:nx nx])+u(:,[1 1:nx-1])-2*u;
uyy = u([2:ny ny],:)+u([1 1:ny-1],:)-2*u;
uxy = (ux([2:ny ny],:)-ux([1 1:ny-1],:))/2;

%Structure tensors
ux2 = gauss(ux.*ux,3*s);
uy2 = gauss(uy.*uy,3*s);
uxuy = gauss(ux.*uy,3*s);

% ux2 = ux.*ux;
% uy2 = uy.*uy;
% uxuy = ux.*uy;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ug = gauss(u, s)
s2 = s*s;
[d1,d2] = size(u);
pad = 2*ceil(sqrt(s2));

u = [repmat(u(:,1),[1, pad]) , u , repmat(u(:,d2),[1, pad])];
u = [repmat(u(1,:),[pad, 1]) ; u ; repmat(u(d1,:),[pad, 1])];

U = fft(u,[],1);
U = fft(U,[],2);

[x, y] = meshgrid(0 : d2+(2*pad)-1, 0 : d1+(2*pad)-1);
x = x/(d2+(2*pad));
y = y/(d1+(2*pad));
G = exp(s2*(cos(2*pi*x) + cos(2*pi*y) - 2));
ug = ifft(G.* U,[],2);
ug = real(ifft(ug,[],1));
ug = ug(pad+[1:d1], pad+[1:d2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lam1, lam2] = getEigVec2Dsym(a, b, c)
d = sqrt((a-c).^2 + 4*b.*b);
lam1 = 0.5*(a+c+d); 
lam2 = 0.5*(a+c-d);

