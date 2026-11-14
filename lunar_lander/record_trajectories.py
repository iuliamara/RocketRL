if not __package__:
    from _entrypoint import package_name

    __package__ = package_name()


from .evaluate_metrics import main


if __name__ == "__main__":
    main(telemetry=True, default_model="ppo_lunar_lander_1m.zip")
