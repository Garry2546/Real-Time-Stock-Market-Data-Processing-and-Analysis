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

### Technologies Used
Python: For simulating the stock market data and implementing the producer and consumer logic.
Apache Kafka: For real-time data streaming, ensuring reliable and scalable data flow.
AWS Boto3 SDK: For interacting with AWS services such as S3.
Amazon S3: For storing the processed data in a scalable and durable manner.
AWS Glue: For automated data cataloging, making data discoverable and usable for analysis.
Amazon Athena: For querying and analyzing the data stored in S3 using SQL.
Key Features
Real-Time Data Streaming: The system captures and streams real-time stock market data using Kafka.
Scalable Data Storage: Data is stored in Amazon S3, which scales automatically with the volume of data.
Automated Data Cataloging: AWS Glue Crawlers automate the discovery and cataloging of data, making it easy to manage and query.
Serverless Data Analysis: Amazon Athena provides a serverless environment for querying and analyzing the data, eliminating the need for infrastructure management.




