img = imread('/scratch/Neuron_Process/img8.jp2');
img = 255 - img(:,:,1);
    %mask = imread('Sparse_mask.tif');
    %img(mask==0) = 0;
    %clear mask;
    %remove background
img1 = removeBg(img);
%enhance the longitudnal structures
img2 = preprocessing(img1,1.5, 200);

%skeletonization (use one of the two approaches)
%fast 
skel = bwmorph(img2>0,'diag');
skel = bwmorph(skel,'dilate');
skel = bwmorph(skel,'fill');
skel = bwmorph(skel,'thin',inf);
%alternate slow but accurate 
%skel = WeightOrderedHomotopicThinning(img);

%segment the skeleton
minimumLength = 10;
skel2 = segmentAndPrune(skel);