if not __package__:
    from _entrypoint import package_name

    __package__ = package_name()


from .train import main


if __name__ == "__main__":
    main(default_reward="precision", default_steps=1_000_000)
