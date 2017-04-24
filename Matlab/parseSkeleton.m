function [T, L, Lnum, J, E] = parseSkeleton(skel)
%Get the labels, end points, and junction points
sz = size(skel);
L = zeros(sz,'uint8'); %labels
Lnum = zeros(1000,1); %size
lcnt = 0;
J = zeros(1000,1); %junctions
jcnt = 0;
E = zeros(1000,1); %endpt
ecnt = 0;
nbr = [-sz(1)-1, -sz(1), -sz(1)+1, -1, 1, sz(1)-1, sz(1), sz(1)+1];
T=cell(0);
for i=1:numel(skel)
    if skel(i) == 0
        %non skel pt
        continue;
    end
     %do not include border pixels
    k = mod(i,sz(1));
    if k==0 || k==1 || i<sz(1) || i > (numel(skel)-sz(1))
        continue;
    end
    if L(i)==0
        lcnt = lcnt+1;
        L(i) = lcnt;
        trc = breadthFirstSearch(skel,i,L,nbr);
        Lnum(lcnt) = numel(trc);
        [T{lcnt}.y,T{lcnt}.x] = ind2sub(sz,trc);
        L(trc) = lcnt;
    end
    s = sum(skel(i+nbr)>0);
    if (s==1)
        ecnt=ecnt+1;
        E(ecnt)=i;
    elseif (s>2)
        jcnt=jcnt+1;
        J(jcnt)=i;
    end
end
J = J(1:jcnt);
E = E(1:ecnt);

function trc = breadthFirstSearch(skel,i,L,nbr)
label = L(i);
trc = zeros(1000,1); m = 1; trc(1)= i;
edges = zeros(1000,2); 
k = 1;
while m<=k
    i = trc(m);
    n = find(skel(i+nbr)>0);
    n = nbr(n)+i; 
    for j=1:numel(n)
        if L(n(j))==0
            L(n(j)) = label;
            k=k+1;
            trc(k) = n(j);
            edges(k,:) = [m, k];
        end
    end
    m=m+1;
end
trc = trc(1:k);