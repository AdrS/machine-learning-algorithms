#!/bin/bash

# Downloads a file if it is not already present.
# Arguments: $1 = URL, $2 = output file
function maybe_download() {
  if ! [ -f $2 ]; then
    curl -o $2 "$1"
  fi
}

# Downloads and extracts a zip file.
# Arguments: $1 = URL, $2 = output directory
function download_zip() {
  maybe_download $1 "$2.zip"
  unzip -n $2.zip -d $2
}

# Downloads and extracts a gzip compressed tar file.
# Arguments: $1 = URL, $2 = output directory
function download_tar_gz() {
  maybe_download $1 "$2.tar.gz"
  mkdir -p $2
  tar --skip-old-files -xf $2.tar.gz --directory $2
}

# Downloads and extracts the files for a Kaggle competition
# Arguments: $1 = name of Kaggle competition
function download_kaggle_competition_files() {
  kaggle competitions download -c $1
  mkdir -p $1
  unzip -n $1.zip -d $1
}

# Classification Datasets
download_zip "https://archive.ics.uci.edu/static/public/2/adult.zip" census-income
echo "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income" > census-income/train.csv
cat census-income/adult.data >> census-income/train.csv
echo "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income" > "census-income/test.csv"
grep -E "^[0-9]+," census-income/adult.test >> "census-income/test.csv"

download_kaggle_competition_files DontGetKicked
download_kaggle_competition_files amazon-employee-access-challenge
download_kaggle_competition_files home-credit-default-risk
download_kaggle_competition_files santander-customer-transaction-prediction
download_kaggle_competition_files ieee-fraud-detection
download_kaggle_competition_files microsoft-malware-prediction
download_kaggle_competition_files vsb-power-line-fault-detection
download_kaggle_competition_files playground-series-s4e1

# Regression Datasets
download_kaggle_competition_files boston-housing

download_tar_gz "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz" california-housing
echo "longitude,latitude,housingMedianAge,totalRooms,totalBedrooms,population,households,medianIncome,medianHouseValue" > california-housing/CaliforniaHousing/data.csv
cat california-housing/CaliforniaHousing/cal_housing.data >> california-housing/CaliforniaHousing/data.csv
