import os
import gc

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser


@dataclass
class XttsTrainerArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    output_path: str = field(
        metadata={"help": "Path to pretrained + checkpoint model"}
    )
    train_csv_path: str = field(
        metadata={"help": "Path to train metadata file"},
    )
    eval_csv_path: Optional[str] = field(
        metadata={"help": "Path to eval metadata file"},
    )
    language: Optional[str] = field(
        default="en",
        metadata={"help": "The language you want to train (language in your dataset)"},
    )

    num_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Epoch"},
    )

    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Mini batch"},
    )

    grad_acumm: Optional[int] = field(
        default=1,
        metadata={"help": "Grad accumulation steps"},
    )

    max_audio_length: Optional[int] = field(
        default=255995,
        metadata={"help": "Max audio length"},
    )

    max_text_length: Optional[int] = field(
        default=200,
        metadata={"help": "Max text length"},
    )

    weight_decay: Optional[float] = field(
        default=1e-2,
        metadata={"help": "Max text length"},
    )

    lr: Optional[float] = field(
        default=5e-6,
        metadata={"help": "Learning rate"},
    )

    save_step: Optional[int] = field(
        default=5000,
        metadata={"help": "Save step"},
    )



def train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length, max_text_length, lr, weight_decay, save_step):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    # OUT_PATH = os.path.join(output_path, "run", "training")
    OUT_PATH = output_path
    SPEAKER_REFERENCES = [
    "/teamspace/studios/this_studio/mosXTTS/reference_1_speaker_male_17.wav",  # speaker reference to be used in training test sentences
    "/teamspace/studios/this_studio/mosXTTS/reference_2_speaker_male_17.wav",
    "/teamspace/studios/this_studio/mosXTTS/reference_3_speaker_male_17.wav"
]

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = True  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps


    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=os.path.basename(train_csv),
        meta_file_val=os.path.basename(eval_csv),
        language=language,
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    #DVAE_CHECKPOINT = "/teamspace/studios/this_studio/mosXTTS/checkpoints/XTTS_v2.0_original_model_files/dvae.pth"
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
    #MEL_NORM_FILE = "/teamspace/studios/this_studio/mosXTTS/models/mel_stats.pth"
    
    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    #TOKENIZER_FILE = "/teamspace/studios/this_studio/mosXTTS/models/vocab.json"
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    #XTTS_CHECKPOINT = "/teamspace/studios/this_studio/mosXTTS/models/best_model_28574.pth"
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files(
            [XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files(
            [XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=11025,  # 0.5 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config

    config = GPTTrainerConfig()

    config.load_json(XTTS_CONFIG_FILE)

    config.epochs = num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = """
        GPT XTTS training
        """,
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 3
    config.save_checkpoints = True
    config.print_eval = True
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay}
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = [
            {
                "text": "La cofr bedre pʋgẽ, naaba gãaga sɛɛga, kamba rogem-pʋɩɩs zalle zalle n puki.",
                "speaker_wav": SPEAKER_REFERENCES[0],
                "language": "mos",
            },
            {
                "text": "A waoongo, a ma kõ la zaabre rɩɩbo. A ri bilfu, la wa sɩnga wɛsla lq ribo. yẽ:",
                "speaker_wav": SPEAKER_REFERENCES[1],
                "language": "mos",
            },
            {
                "text": "Pʋg-sɛda, barka. Sẽn na n yif sõngre, mam nãa n kõnf la sagleg sõngo. Fo sãa n wa be yelpakre pʋgẽn, tẽng zikãnga yele la ef wa yãama.",
                "speaker_wav": SPEAKER_REFERENCES[2],
                "language": "mos",
            },
        ]

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS
        ),
        config,
        output_path=os.path.join(output_path, "run", "training"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(str(item["text"]).split(" ")) for item in train_samples]
    longest_text_idx =  samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return trainer_out_path


if __name__ == "__main__":
    parser = HfArgumentParser(XttsTrainerArguments)

    args = parser.parse_args_into_dataclasses()[0]

    trainer_out_path = train_gpt(
        language=args.language,
        train_csv=args.train_csv_path,
        eval_csv=args.eval_csv_path,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_acumm,
        weight_decay=args.weight_decay,
        lr=args.lr,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        save_step=args.save_step
    )

    print(f"Checkpoint saved in dir: {trainer_out_path}")
