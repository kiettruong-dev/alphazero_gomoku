# -*- coding: utf-8 -*-
"""
Cau hinh logging dung chung cho self-play va training.

Cach dung:
    from logging_setup import setup_logging
    logger = setup_logging(level=logging.DEBUG, console_level=logging.INFO)

- level=logging.DEBUG  -> ghi toan bo chi tiet tung playout (P, Q, u, expand, backup) vao FILE log.
- console_level=logging.INFO -> man hinh chi hien thi tom tat tung nuoc di / tung epoch train,
  khong bi tran man hinh boi hang tram playout.

Neu muon xem CA chi tiet playout tren console luon (khong chi trong file),
truyen console_level=logging.DEBUG (se rat nhieu dong, chi nen dung khi n_playout nho, vi du 5-10).
"""

import logging
import os
import datetime

LOGGER_NAME = "alphazero"


def setup_logging(log_dir="logs", level=logging.DEBUG, console_level=logging.INFO,
                   log_filename=None):
    """
    Khoi tao logger 'alphazero' voi 2 handler:
      - FileHandler: ghi toan bo (mac dinh DEBUG) vao file, dung de xem lai sau.
      - StreamHandler: in ra console theo console_level (mac dinh INFO, tom tat).
    Goi ham nay MOT LAN duy nhat luc bat dau chuong trinh (vi du dau train.py).
    """
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = "selfplay_{}.log".format(ts)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # logger goc luon mo DEBUG, handler moi ben quyet dinh loc gi
    logger.handlers.clear()  # tranh nhan doi handler neu goi lai nhieu lan

    fmt_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    fmt_console = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    logger.propagate = False
    logger.info("=== Bat dau ghi log. File log chi tiet: %s ===", log_path)
    return logger


def get_logger():
    """Lay logger da tao (goi setup_logging truoc, o day chi lay lai instance)."""
    return logging.getLogger(LOGGER_NAME)


def setup_worker_logging(worker_tag, log_dir="logs/selfplay_workers", level=logging.DEBUG):
    """
    Cau hinh logging danh RIENG cho 1 tien trinh con (worker) chay self-play song song.

    Khac voi setup_logging():
      - KHONG gan StreamHandler (console) -> tien trinh con se KHONG in
        tung nuoc di / tung playout ra terminal chung, tranh chay chu
        khi chay nhieu tien trinh cung luc.
      - Moi worker ghi ra 1 file RIENG (ten file gan voi worker_tag,
        vi du PID hoac so thu tu van), de neu can xem lai chi tiet 1
        van cu the thi mo dung file, khong bi tron lan voi cac worker khac.

    Goi ham nay 1 LAN o DAU moi tien trinh con (trong _self_play_worker),
    KHONG goi setup_logging() (ham danh cho tien trinh chinh/console).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "worker_{}.log".format(worker_tag))

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    logger.propagate = False
    return logger