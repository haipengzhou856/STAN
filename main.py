import yaml
from utils.utils import *
import accelerate
from accelerate import DistributedDataParallelKwargs
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate.utils import set_seed

if __name__ == '__main__':
    with open(os.path.join("configs/ViShaVideo_STEDiff.yml"), "r") as f: # change the config here for
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    # check out_dir
    time_stamp = time.strftime("%m%d%H%M", time.localtime())
    # add timestamp to distinguish
    # exp: /output/visda/exp09121200/tb_path ##
    out_dir = config.OUTPUT.HOME + config.OUTPUT.DATA_NAME + "/" + config.OUTPUT.MODEL_NAME + time_stamp
    tb_path = check_dir(out_dir + config.OUTPUT.TB)  # tensorboard
    ckpt_path = check_dir(out_dir + config.OUTPUT.CKPT)  # checkpoint
    log_path = check_dir(out_dir + config.OUTPUT.LOG)  # logging

    result_path = check_dir(out_dir + config.OUTPUT.RESULT)  # store the test results
    # copy_folder_without_images("/home/haipeng/Code/Data/ViSha/test/labels", result_path)
    # copy the folder name for save results, avoiding process preemption in acceleration when mkdir

    set_seed(config.SEED)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs],gradient_accumulation_steps=2)
    writer = SummaryWriter(tb_path)
    # logger init
    logger = setup_logger(config.OUTPUT.DATA_NAME,
                          log_path,
                          accelerator.process_index,
                          "log.txt")

    # init env
    logger.info("----------------------NEW RUN----------------------------")
    logger.info("----------------------Basic Setting----------------------------")
    logger.info("---work place in: {dir}-----".format(dir=out_dir))
    logger.info("Img_size: {}".format(config.DATASET.IMG_SIZE))
    logger.info("TIME_CLIP: {}".format(config.DATASET.TIME_CLIP))
    logger.info("BATCH_SIZE: {}".format(config.DATASET.BATCH_SIZE))
    logger.info("lr: {}".format(config.SOLVER.LR))
    logger.info("opim: {}".format(config.SOLVER.OPTIM))

    logger.info("----------------------Diffusion----------------------------")
    logger.info("timestep: {}".format(config.DIFFUSION.TIMESTEPS))
    logger.info("BitScale: {}".format(config.DIFFUSION.SCALE))
    logger.info("Scheduler: {}".format(config.DIFFUSION.SCHEDULER))
    logger.info("TimeDifference: {}".format(config.DIFFUSION.TD))
    logger.info(
        "--------------------USE {model_name}-----------------------".format(model_name=config.OUTPUT.MODEL_NAME))
    logger.info(
        "Using {num_gpu} GPU for training, {mix_pix} mix_precision used.".format(num_gpu=accelerator.num_processes,
                                                                                 mix_pix=accelerator.mixed_precision))
    model_name = config.OUTPUT.MODEL_NAME
    
    
    from engine.trainer import training_func
    if "PEDiff" in model_name:
        from models.PEDiff import VideoSegformer, Segformer

        pretrain_model = Segformer()
        model = VideoSegformer(PretrainedSegformer=pretrain_model,
                               bit_scale=config.DIFFUSION.SCALE,
                               timesteps=config.DIFFUSION.TIMESTEPS,
                               noise_schedule=config.DIFFUSION.SCHEDULER,
                               time_difference=config.DIFFUSION.TD,
                               num_frames=config.DATASET.TIME_CLIP)

    elif "Pix2Seq" in model_name:
        from models.Pix2Seq import VideoSegformer, Segformer

        pretrain_model = Segformer()
        model = VideoSegformer(PretrainedSegformer=pretrain_model,
                               bit_scale=config.DIFFUSION.SCALE,
                               timesteps=config.DIFFUSION.TIMESTEPS,
                               noise_schedule=config.DIFFUSION.SCHEDULER,
                               time_difference=config.DIFFUSION.TD,
                               num_frames=config.DATASET.TIME_CLIP)

    elif "STEDiff" in model_name:
        from models.STEDiff import VideoSegformer, Segformer

        pretrain_model = Segformer()
        model = VideoSegformer(PretrainedSegformer=pretrain_model,
                               bit_scale=config.DIFFUSION.SCALE,
                               timesteps=config.DIFFUSION.TIMESTEPS,
                               noise_schedule=config.DIFFUSION.SCHEDULER,
                               time_difference=config.DIFFUSION.TD,
                               num_frames=config.DATASET.TIME_CLIP)           
              
        assert "NO MODEL IMPLEMENTED"

    model = model.to(device=accelerator.device)
    training_func(config, accelerator, model, logger, writer, ckpt_path, result_path)

    # TODO
    # eval()

    logger.info("----------------------END RUN----------------------------")

