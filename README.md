# Stock-Market-Analysis

### Project Overview
This project demonstrates a real-time stock market data processing and analysis system using modern data engineering and cloud technologies. The architecture captures real-time stock data, processes it, and stores it for further analysis using Amazon Web Services (AWS). The system leverages Apache Kafka for data streaming, Amazon S3 for storage, AWS Glue for data cataloging, and Amazon Athena for querying the stored data.

## Architecture Description
### Data Source (Stock Market App Simulation):

A Python-based stock market simulation app fetches real-time stock data from APIs.

Producer:

The simulated stock data is sent to Apache Kafka, running on an Amazon EC2 instance. Kafka acts as the data streaming platform, providing a scalable and fault-tolerant way to capture and transport the data.

### Kafka (Running on Amazon EC2):

Kafka is used for real-time data streaming. The producer sends stock market data to Kafka topics, and a consumer fetches the data for storage and analysis.
Consumer:

The Kafka consumer reads the real-time data from the Kafka topics and processes it.

### Amazon S3:

Processed data is stored in Amazon S3, a scalable object storage service. This allows for durable storage of the stock market data, making it available for future analysis and machine learning workflows.

### AWS Glue Data Catalog:

AWS Glue Crawler scans the data in S3 and automatically populates the AWS Glue Data Catalog. This catalog acts as a metadata repository, storing schema information about the data stored in S3.
### Amazon Athena:

Amazon Athena is used to query the data stored in S3. Athena allows for serverless, ad-hoc analysis of the data using standard SQL queries, leveraging the metadata stored in the AWS Glue Data Catalog.


![Architecture](https://github.com/user-attachments/assets/f59dd564-120f-4bc8-933a-ef02bd0cfd5d)



