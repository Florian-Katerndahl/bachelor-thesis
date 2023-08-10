#! /bin/bash

set -xe

OLD_limit=$(ulimit -s)
ulimit -s 65536

function pre_process_tiles() {
	local BASE_NAME=$(basename $1)

	wget --quiet --content-disposition "${BASE_URL}/${dataset}.${SUFFIX}"

	unzip -qq "${dataset}.${SUFFIX}" -d "${BASE_NAME}"

	mv "${dataset}.${SUFFIX}" raw-data

	docker run --rm -u$(id -u):$(id -g) -v $PWD/${BASE_NAME}:/data floriankaterndahl/ecw2tiff \
		/bin/bash -c $'find /data -maxdepth 1 -name \'*ecw\' -print0 | xargs -0 -I{} bash -c \'gdal_translate -ot "Byte" -a_srs "EPSG:25833" -co "COMPRESS=LZW" -co "PREDICTOR=2" $1 $(sed "s/ecw/tif/" <<< $1)\' bash {}'

	mv $PWD/${BASE_NAME}/*tif regions/

	rm -r "${BASE_NAME}"

	return 0
}

BASE_URL="https://fbinter.stadt-berlin.de/fb/atom/DOP/dop20true_rgb_2020"
REGIONS=("Mitte" "Nord" "Nordost" "Nordwest" "Ost" "Sued" "Suedost" "Suedwest" "West")
SUFFIX="zip"
N_DATASETS=${#REGIONS[*]}
PROCESSED=0

if [ ! -d "raw-data" ]; then
  mkdir "raw-data"
fi

if [ ! -d "regions" ]; then
	mkdir "regions"
fi

if [ ! -d "tiles" ]; then
	mkdir "tiles"
fi

for dataset in "${REGIONS[@]}"; do
	(pre_process_tiles "${dataset}" && echo "Done processing ${dataset}")&
done

echo "Waiting for parallel processes to finish"

wait $(jobs -p)

echo "Done"

find "./regions" -name "*tif" -type f -execdir gdal_retile.py -ps 512 512 -overlap 0 -ot "Byte" -r "near" -s_srs "EPSG:25833" -targetDir "../tiles" {} +

find "./tiles" -type f -name "*.tif" -fprint tif_names.txt

cat tif_names.txt | xargs -I{} rename 's/([a-zA-Z0-9]+)_\d{3}_\d{4}_\d_be_\d{4}_(\d+)_(\d+).tif/$1_Berlin_$2_$3.tif/' {}

find "./tiles" -maxdepth 1 -type f -name "*tif" -exec /bin/bash -c 'convert $0 .$(cut -d . -f1 <<< "$0").png' {} \;

find "./tiles" -type f -fprint vrt_list.txt

gdalbuildvrt -input_file_list vrt_list.txt output.vrt

ulimit -s $OLD_LIMIT
