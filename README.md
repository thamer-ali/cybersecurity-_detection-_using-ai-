 converting cvs to paquet 
 
 The script is a PySpark program designed to convert a large CSV network dataset into Parquet format while tracking the execution process through a monitoring log.

It performs the following steps:

Initialize Apache Spark
A Spark session is created to enable distributed processing of the large dataset.
Define input and output paths
The script reads the CSV file from /data/august.week4.csv.uniqblacklistremoved and writes the output to /data/parquet/august_week4.
Implement a monitoring mechanism
A logging function records all execution steps into a file (/data/parquet_conversion_monitor.log) with timestamps.
Validate input data
The script checks whether the input file exists and logs its size.
Load the CSV dataset
The dataset is read into a Spark DataFrame.
Convert and save as Parquet
The data is written in Parquet format using distributed processing.
Log execution status
The script records key events such as start, progress, successful completion, or errors.
Terminate Spark session
The Spark session is safely stopped after execution.
