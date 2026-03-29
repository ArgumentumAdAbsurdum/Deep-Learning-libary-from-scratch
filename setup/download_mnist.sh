#!/bin/bash
mkdir -p datasets
cd datasets

URL="https://github.com/phoebetronic/mnist/raw/main/mnist_train.csv.zip"
FILE="mnist_train.csv.zip"

URL2="https://github.com/phoebetronic/mnist/raw/main/mnist_test.csv.zip"
FILE2="mnist_test.csv.zip"

if [ ! -f "mnist_train.csv" ]; then
    curl -L -O "$URL"
    unzip -o "$FILE"
    rm "$FILE"
else
    echo "Die Datei mnist_train.csv existiert bereits im Ordner 'datasets'."
fi



if [ ! -f "mnist_test.csv" ]; then
    curl -L -O "$URL2"
    unzip -o "$FILE2"
    rm "$FILE2"
else
    echo "Die Datei mnist_test.csv existiert bereits im Ordner 'datasets'."
fi

