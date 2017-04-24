function skel = WeightOrderedHomotopicThinning(H)
sz = size(H);
skel = zeros(sz,'uint8');
skel(H>0) = 1;
skel(:,1) = 0;
skel(:,end) = 0;
skel(1,:) = 0;
skel(end,:) = 0;

nbr = [-sz(1)-1, -sz(1), -sz(1)+1, -1, 1, sz(1)-1, sz(1), sz(1)+1];
N = find(skel);
heap = zeros(numel(N),2) + inf;
heapRec = zeros(sz,'int8');
hndx = 1;
for k=1:numel(N)
    n = N(k);
    if any(skel(n+nbr)==0)
        heap(hndx,:) = [n, H(n)];
        heapRec(n) = 1;
        hndx = hndx+1;
    end
end

display = 0;

while hndx > 1
    %pop the min element
    heap = sortrows(heap,2); 
    h = heap(1,1);
    %fprintf('ndx = %d value = %0.3f\n',hndx, heap(1,2));
    heap(1,:) = inf; 
    if display
        P = zeros(sz,'uint8');
        P(H>0) = 128; P(h) = 255;
        im([P, skel*255]); pause(0.3);
    end
    
    nh = h+nbr;
    if IsSimple(skel(nh))==1
        skel(h) = 0;
        for i=1:8
            if skel(nh(i)) == 0
                continue;
            end
            if (heapRec(nh(i)) == 1)
                continue;
            end
            if any(skel(nh(i)+nbr)==0)
                hndx = hndx+1;
                heap(hndx,:) = [nh(i), H(nh(i))];
                heapRec(nh(i)) = 1;
            end
        end
    end
    hndx = hndx-1;
end

function F = IsSimple(nbr)
B = zeros(3,3,'uint8');
B(1:4) = nbr(1:4);
B(6:9) = nbr(5:8);
%use Euler formula to get C + V - E = F 
V = sum(B(:)==1);
Elist = [1,2; 2,3; 1,4; 2,4; 2,6; 3,6; ...
            7,8; 8,9; 7,4; 8,4; 8,6; 9,6];
Esum  = B(Elist(:,1)) + B(Elist(:,2)) ;
E = sum(Esum==2);
Clist = [1, 2, 4; 2 3 6; 4 7 8; 8 9 6];
Csum = B(Clist(:,1)) + B(Clist(:,2)) + B(Clist(:,3));
C = sum(Csum==3);
F = V-E+C;
%end point
if (V==2 && E==1) || (V==1)
    F = 0;
end
% if F ~= 1 && F ~= 2
%     disp(B);
%     fprintf('F = %d\n',F); 
% end
