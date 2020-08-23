import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Tool to help the creation of TanD projects")
    parser.add_argument('--init-git', action='store_true', help='if passed, this flags sets the deployment script to '
                                                                'start a git repo on this directory, adding all the '
                                                                'files to it')
    args = parser.parse_args()

    folder = '.ebextensions'
    try:
        os.mkdir(folder)
    except:
        print(f"{folder} folder already exists")

    config_env_variables = """option_settings:
  aws:elasticbeanstalk:application:environment:
    MLFLOW_TRACKING_URI: sqlite:///database.db
    MLFLOW_DEFAULT_ARTIFACT_ROOT: ./mlruns/
    API_TOKEN: TOKEN123
    MODEL_STAGE: Production"""

    with open(os.path.join(folder, "options.config"), "w") as file:
        file.write(config_env_variables)
        file.close()

    config_scaling = """option_settings:
  aws:autoscaling:asg:
    MinSize: 2
    MaxSize: 10
  aws:ec2:instances:
    InstanceTypes: t2.micro,t3.micro
  aws:autoscaling:trigger:
    BreachDuration: 5
    LowerBreachScaleIncrement: -1
    LowerThreshold: 1.0
    MeasureName: Latency
    Period: 5
    EvaluationPeriods: 1
    Statistic: Average
    Unit: Seconds
    UpperBreachScaleIncrement: 1
    UpperThreshold: 10"""

    with open(os.path.join(folder, "scaling.config"), "w") as file:
        file.write(config_scaling)
        file.close()

    if args.init_git:
        deployment_command = """
        git init && git add . && git commit -m "preparing to deploy on aws-eb"
        
        eb init -p docker tand-api-project && eb create tand-api-project-env
        """
    else:
        deployment_command = """
        eb init -p docker tand-api-project && eb create tand-api-project-env
        """

    with open("deploy-aws-eb.sh", "w") as file:
        file.write(deployment_command)

    cron_config = """files:
    "/etc/cron.d/mycron":
        mode: "000644"
        owner: root
        group: root
        content: |
            0 0 * * * root /usr/local/bin/myscript.sh

    "/usr/local/bin/myscript.sh":
        mode: "000755"
        owner: root
        group: root
        content: |
            #!/bin/bash

            # Your actual script content
            source /opt/elasticbeanstalk/deployment/env.list && curl -H "Content-Type: application/json" -H "TOKEN: $API_TOKEN" -X POST http://localhost/update-model

            exit 0

commands:
    remove_old_cron:
        command: "rm -f /etc/cron.d/mycron.bak"
    
    """

    with open(os.path.join(folder, "cron.config"), "w") as file:
        file.write(cron_config)
        file.close()


if __name__ == '__main__':
    main()
