#!/bin/bash


# Check if the tag value is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <tag-value>"
  exit 1
fi

TAG_VALUE=$1

INSTANCE_IP=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=$TAG_VALUE" --query "Reservations[*].Instances[*].PublicIpAddress" --output text)



# Check if the IP address was fetched successfully
if [ -z "$INSTANCE_IP" ]; then
  echo "No instance found with the tag 'Name=sache' or instance does not have a public IP address."
  exit 1
fi

echo "Fetched IP address: $INSTANCE_IP"

# Use the fetched IP address in the scp commands

scp -i ~/.ssh/id_sache -o StrictHostKeyChecking=no /Users/plato/.ssh/id_remotekey ubuntu@$INSTANCE_IP:~/.ssh/id_rsa
if [ $? -ne 0 ]; then
    echo "unable to connect"
    exit 1
fi

scp -i ~/.ssh/id_sache -o StrictHostKeyChecking=no /Users/plato/.ssh/id_remotekey.pub ubuntu@$INSTANCE_IP:~/.ssh/id_rsa.pub

ssh -i ~/.ssh/id_sache ubuntu@$INSTANCE_IP 'chmod 600 ~/.ssh/id_rsa'
echo "Changed permissions for id_rsa on the EC2 instance."

ssh -i ~/.ssh/id_sache ubuntu@$INSTANCE_IP 'ssh-keyscan -H github.com >> ~/.ssh/known_hosts'
ssh -i ~/.ssh/id_sache ubuntu@$INSTANCE_IP 'git clone git@github.com:lewington-pitsos/sae_vit'

scp -i ~/.ssh/id_sache -o StrictHostKeyChecking=no .credentials.json ubuntu@$INSTANCE_IP:~/sae_vit/
echo "Copied credentials over"

sed -i '' "/^Host aws-sache$/,/^$/ s/HostName .*/HostName $INSTANCE_IP/" /Users/plato/.ssh/config


ssh -i ~/.ssh/id_sache ubuntu@$INSTANCE_IP 'cd ~/sae_vit && pip install boto3 wandb && python scripts/download_logs.py'
