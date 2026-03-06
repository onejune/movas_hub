dtm=$1
sample_path="oss://spark-ml-train-new/liufashuai/schedule/databack/data/parquet/$dtm/"

ossutil cp $sample_path ./data/ --recursive