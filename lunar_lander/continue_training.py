if not __package__:
    from _entrypoint import package_name

    __package__ = package_name()


from .config import MODEL_DIR
from .train import main


if __name__ == "__main__":
    main(default_steps=700_000, default_resume=MODEL_DIR / "ppo_lunar_lander.zip")
