img = imread('2.png');
mask = imread('2_mask.png');

mask = imdilate(mask, strel(ones(5,5))); 
mask = mask/max(mask(:));

%img = medfilt2(img);

bw = sauvola(img);

bw = bwmorph(bw,'skel',Inf);
CC = bwconncomp(bw);
numOfPixels = cellfun(@numel,CC.PixelIdxList);
[unused,indexOfMax] = max(numOfPixels);
bw2 = zeros(size(bw));
bw2(CC.PixelIdxList{indexOfMax}) = 1;

bw2 = double(bw2) .* double(mask);

bw2 = bwmorph(bw2,'spur',Inf);
bw2 = bwmorph(bw2,'clean');

imshow(bw2);
imwrite(bw2,'2_bw.png');
