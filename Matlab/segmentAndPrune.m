function skel2 = segmentAndPrune(skel)
sz = size(skel);
junctions = zeros(sz(1),sz(2),'uint8');
nbr = [-sz(1)-1, -sz(1), -sz(1)+1, -1, 1, sz(1)-1, sz(1), sz(1)+1];
for i=2:sz(1)-1
    for j=2:sz(2)-1
        n = j+sz(1)*(i-1);
        if skel(n)>0
            nn = skel(nbr+n)>0;
            if sum(nn)>2
                skel(n) = 0;
                junctions(n) = 255;
            end
        end
    end
end
CCskel = bwconncomp(skel);
L = labelmatrix(CCskel);
CCjunctions = bwconncomp(junctions);
clear junctions;
L2 = zeros(sz(1),sz(2),'uint8');
for i=1:CCjunctions.NumObjects
    c = 0;
    Cand = zeros(10,4);
    for j=1:numel(CCjunctions.PixelIdxList{i})
        N = find(CCjunctions.PixelIdxList{i}(j)+nbr);
        for k=1:8
            if L(N(k))>0
                d = findDir(N(k),L);
                c = c+1;
                Cand(c,:) = [d(1) d(2) N(k) L(N(k))];
            end
        end
    end
    CC = ones(c,c);
    for c1=1:c
        for c2=1:c
            CC(c1,c2) = Cand(c1,1)*Cand(c2,1) + Cand(c1,2)*Cand(c2,2);
        end
    end
    [~, cord] = sort(CC(:));
    for b=1:c
        cc = cord(b);
        [c1,c2] = ind2sub([c c],cc);
        if Cand(c1,4) ~= Cand(c2,4)
            %all junction and should be assigned same label
            break;
        end
    end
end

function d = findDir(n,L, sz)
nbr = [-sz(1)-1, -sz(1), -sz(1)+1, -1, 1, sz(1)-1, sz(1), sz(1)+1];
lbl = L(n);
T = zeros(20,1); T(1)= n; t = 0;
for k=1:10
    n1 = n+nbr;
    b = find(L(n1)==lbl);
    if numel(b)
        break;
    else
        T(t+1:numel(b)) = n1(b);
        t = t+numel(b);
    end
end
[y,x] = ind2sub(sz,T(1:t));
y = y-y(1); x = x-x(1);
dn = sqrt(x.*x + y.*y)+1e-2;
[~,m] = max(dn);
d = [x(m)/dn(m) y(m)/dn(m)];

