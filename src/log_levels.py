"""Adds SUCCESS and FAIL log levels."""
import logging

SUCCESS = 25
FAIL = 35

if not hasattr(logging.Logger, "success"):
    logging.addLevelName(SUCCESS, "SUCCESS")
    logging.addLevelName(FAIL, "FAIL")

    def _success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, **kwargs)

    def _fail(self, msg, *args, **kwargs):
        if self.isEnabledFor(FAIL):
            self._log(FAIL, msg, args, **kwargs)

    logging.Logger.success = _success
    logging.Logger.fail = _fail
