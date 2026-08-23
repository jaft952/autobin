"""
src/log_levels.py

Registers SUCCESS/FAIL as real logging levels so any module can call
log.success(...) / log.fail(...) like the built-in log.info()/log.warning().
Import this before using either — arm_executor.py and web/logbuffer.py both
do, so it's set up regardless of which one runs first.

DEBUG(10) < INFO(20) < SUCCESS(25) < WARNING(30) < FAIL(35) < ERROR(40)
"""
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
