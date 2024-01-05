#!/bin/sh
# go to the dataset location

# altering the zip files

zip -F L.zip --out L_Scale.zip
zip -FF L_Scale.zip --out L.zip -fz
zip -F M.zip --out M_Scale.zip
zip -FF M_Scale.zip --out M.zip -fz

rm -f L_Scale.zip
rm -f L.z01
rm -f M_Scale.zip
rm -f M.z01
rm -f M.z02

# build a directory of datasets
mkdir datasets
mv L.zip datasets
mv M.zip datasets
mv S.zip datasets

cd datasets
unzip L.zip
unzip M.zip
unzip S.zip

rm -f L.zip
rm -f M.zip
rm -f S.zip

mkdir All
cp -r L/* All/ &
cp -r M/* All/ &
cp -r S/* All/