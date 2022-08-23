# example
scenario_name="academy_pass_and_shoot_with_keeper"

python main.py --config=rode --env-config=football
    with env_args.scenario_name=${scenario_name} \
    test_interval=2000 test_nepisodes=20

python main.py --config=rode --env-config=mpe \
    with env_args.scenario_name=${scenario_name} \
    test_interval=2000 test_nepisodes=20
