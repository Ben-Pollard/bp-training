from pytorch_lightning.callbacks import ModelCheckpoint


class MLFlowNoSaveModelCheckpoint(ModelCheckpoint):
    """
    Class to override checkpoint saving behaviour of Lightning.
    we want mlflow to handle checkpoint saving, so we allow
    lightning to track metrics but prevent it from saving.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        # trainer.save_checkpoint(filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
