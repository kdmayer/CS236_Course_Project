### Google Cloud Platform

To identify available machines on Google Cloud Platform, run the following command:

```cloud compute machine-types list --filter="us-west1-b" | grep gpu```

### AWS

Navigate to your aws_key folder to connect to your VM with:

```ssh -i "cs236-key.pem" ubuntu@ec2-3-19-66-237.us-east-2.compute.amazonaws.com```