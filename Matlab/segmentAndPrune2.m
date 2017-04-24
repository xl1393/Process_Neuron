function epts = segmentAndPrune2(img,skel)
sz = size(skel);
EnJ = zeros(sz(1),sz(2),'uint8'); %endpoint and junctions
epts = zeros(100000,3); endx = 0;
nbr = [-sz(1)-1, -sz(1), -sz(1)+1, -1, 1, sz(1)-1, sz(1), sz(1)+1];
for iter=1:5
    for i=2:sz(1)-1
        for j=2:sz(2)-1
            n = j+sz(1)*(i-1);
            if skel(n)>0
                nn = sum(skel(nbr+n)>0);
                if nn>2
                    EnJ(n) = 3;
                elseif nn==1
                    endx = endx+1;
                    epts(endx,1) = n;
                    EnJ(n) = 1;
                elseif nn==2
                    EnJ(n) = 2;
                end
            end
        end
    end
    
    for e=1:endx
        n = epts(e,1);
        n1 = EnJ(n+nbr)==2;
        val = img(n); cnt = 1;
        while sum(n1)==1
            n0 = n;
            n = n+nbr(n1);
            val = val+img(n); cnt = cnt+1;
            nn = n+nbr;
            n1 = (EnJ(nn)==2) & (nn ~= n0);
        end
        if EnJ(n)==1
            val = val+img(n); cnt = cnt+1;
            %fprintf('%d.',n);
        end
        epts(e,2) = val/cnt;
        epts(e,3) = cnt;
        fprintf('Avg %d %f\n',cnt,val/cnt);
    end
    thresh = mean(epts(1:endx,2));
    for e=1:endx
        if epts(e,2) < thresh
            n = epts(e,1);
            n1 = EnJ(n+nbr)==2;
            while sum(n1)==1
                skel(n)=0;
                n0 = n;
                n = n+nbr(n1);
                nn = n+nbr;
                n1 = (EnJ(nn)==2) & (nn ~= n0);
            end
        end
    end
end
