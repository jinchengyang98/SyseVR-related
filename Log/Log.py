# log_config.py
import logging
from config.Trainingconfig import TrainingConfig  # 训练配置
config = TrainingConfig()
from datetime import datetime
def setup_logging(task_mode = config.task_name):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        handlers=[
                            logging.FileHandler(config.log_path + "_"+ config.task_name + "_" + timestamp , mode='a'),
                            logging.StreamHandler()
                        ])

