mkdir -p data && cd $_
wget -c http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz # All Images and Annotations, 1.1 gigabytes
wget -c http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz # Segmentations, 37MB
tar -xvf CUB-200-2011.tgz
tar -xvf segmentations.tgz
cd ..