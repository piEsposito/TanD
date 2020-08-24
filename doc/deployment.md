# TanD.deployment module

# Index:
  * [aws_elastic_beanstalk.main](#func-aws_elastic_beanstalk.main)
  
## func aws_elastic_beanstalk.main
### tand.deployment.aws_elastic_beanstalk.main()

Generate deploy-aws-eb.sh, which will be run for deployment. It will also generate .ebextensions containing:

    cron.config - which runs, on each instance, a daily task to update the instance ML model by fetching the last production one from mlflow (which is properly used when we set cloud-based mlflow backend);
    options.config - which sets the API token and mlflow backend env variables for the deployment; and
    scaling.config - which sets the scalability configurations for the deployment, including the maximum and minimum number of replicas and criteria for scaling (defaults to latency)

CLI parameters:
 * `--init-git` - If passed, includes git repository init and adding whole directory to it, in order to help `eb` cli deployment.