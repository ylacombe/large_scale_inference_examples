import logging
import sys
import os
import re


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
from tqdm import tqdm

import datasets
import torch
import transformers

from datasets import load_dataset, Audio, concatenate_datasets, load_from_disk, Dataset
from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, HfArgumentParser
from torch.utils.data import DataLoader

@dataclass
class InferenceArguments:
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
    dataloader_num_workers: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for inference data loading (PyTorch only). 0 means that the data will be loaded in the main process."
            )
        },
    )
    output_dir: str = field(
        default=None,
        metadata={
            "help": (
                "The output directory where the datasets will be written."
            )
        },
    )
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": (
            "Random seed to be used with data samplers. If not set, random generators for data sampling will use the"
            "same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model"
            "seed."
            )
        },
    )
    per_device_batch_size: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "Batch size per GPU/TPU/MPS/NPU core/CPU for inference."
            )
        },
    )
    
    

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for large scale inference.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of a dataset from the datasets package"})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split_name: str = field(
        default="train",
        metadata={
            "help": (
                "The name of the inference data set split to use (via the datasets library). Defaults to 'train'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    max_length_seconds: float = field(
        default=20,
        metadata={"help": "Audio clips will be randomly cut to this length during training if the value is set."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    temporary_save_to_disk: str = field(default=None, metadata={"help": "Temporarily save audio labels here."})
    save_data_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Temporarily save the audio labels every `save_data_steps`."},
    )
    remove_audio_column: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to remove the audio column. Helps saving the final dataset much faster."
            )
        },
    )
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch or
    to `max_length` if `max_length` is set and `padding=max_length`.
    """

    feature_extractor: AutoFeatureExtractor
    audio_column_name: str
    feature_extractor_input_name: Optional[str] = "input_values"
    max_length: Optional[int] = None
    padding: Optional[str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        audios = [feature[self.audio_column_name]["array"] for feature in features]
        len_audio = [len(audio) for audio in audios]
        if self.max_length is not None:
            audios = [audio[: min(l, self.max_length)] for audio, l in zip(audios, len_audio)]

        # since resampling has already been performed in the 'load_multiple_datasets' function,
        # a fixed sampling_rate is passed to the feature_extractor.
        sampling_rate = self.feature_extractor.sampling_rate
        batch = self.feature_extractor(
            audios, sampling_rate=sampling_rate, return_tensors="pt", padding=self.padding
        )
        return batch


CHECKPOINT_CODEC_PREFIX = "checkpoint"
_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")

def save_checkpoint(output_dir, dataset, step, num_proc=0):
    checkpoint_path = f"{CHECKPOINT_CODEC_PREFIX}-{step}"
    output_path = os.path.join(output_dir, checkpoint_path)
    dataset.save_to_disk(output_path, num_proc=num_proc)


def load_checkpoint(checkpoint_path):
    dataset = load_from_disk(checkpoint_path)
    return dataset


def sorted_checkpoints(output_dir=None) -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{CHECKPOINT_CODEC_PREFIX}-*")]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{CHECKPOINT_CODEC_PREFIX}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def load_all_checkpoints(output_dir=None) -> List[str]:
    """Helper function to load and concat all checkpoints."""
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir)
    datasets = [load_from_disk(checkpoint) for checkpoint in checkpoints_sorted]
    datasets = concatenate_datasets(datasets, axis=0)
    return datasets


def get_last_checkpoint_step(folder) -> int:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return 0
    content = os.listdir(folder)
    checkpoints = [path for path in content if _RE_CHECKPOINT.search(path) is not None]
    if len(checkpoints) == 0:
        return 0
    last_checkpoint = os.path.join(
        folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0]))
    )
    # Find num steps saved state string pattern
    pattern = r"checkpoint-(\d+)"
    match = re.search(pattern, last_checkpoint)
    cur_step = int(match.group(1))
    return cur_step



logger = logging.getLogger(__name__)


def main():
    # See all possible arguments by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, inference_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, inference_args = parser.parse_args_into_dataclasses()

    if inference_args.dtype == "float16":
        mixed_precision = "fp16"
    elif inference_args.dtype == "bfloat16":
        mixed_precision = "bf16"
    else:
        mixed_precision = "no"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        project_dir=inference_args.output_dir,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {accelerator.process_index}, device: {accelerator.device}, n_processes: {accelerator.num_processes}, "
        f"distributed training: {accelerator.use_distributed}, mixed precision: {accelerator.mixed_precision}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Test all gather - used for warmout and avoiding timeout
    logger.debug(str(accelerator.process_index), main_process_only=False, in_order=True)
    test_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered_tensor = accelerator.gather(test_tensor)
    print("gathered_tensor", gathered_tensor)
    accelerator.wait_for_everyone()


    # Set seed before initializing model.
    set_seed(inference_args.seed)
    num_workers = data_args.preprocessing_num_workers


    # 1. First, let's instantiate the feature extractor, tokenizers and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    sampling_rate = feature_extractor.sampling_rate
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # audio classification related
    id2label = model.config.id2label

    # 2. Now, let's load the dataset
    with accelerator.local_main_process_first():
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.dataset_split_name,
            num_proc=num_workers,
        )

        if sampling_rate is not None and data_args.audio_column_name is not None:
            # resample target audio
            dataset = dataset.cast_column(data_args.audio_column_name, Audio(sampling_rate=sampling_rate))


    ####### B. Classify audio

    logger.info(f"*** Classify audio with {model_args.model_name_or_path} ***")

    # no need to prepare audio_decoder because used for inference without mixed precision
    # see: https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.prepare
    model = accelerator.prepare_model(model, evaluation_mode=True)


    data_collator = DataCollatorWithPadding(
        feature_extractor,
        audio_column_name=data_args.audio_column_name,
        feature_extractor_input_name=feature_extractor_input_name,
        max_length=data_args.max_length_seconds*sampling_rate,
    )

    def apply_audio_decoder(batch):
        with torch.no_grad():
            logits = model(**batch).logits
            scores = torch.nn.functional.softmax(logits, dim=-1)
            pred = torch.argmax(scores, dim=1)
        
        output = {}
        output["labels"] = pred 

        return output


    data_loader = DataLoader(
        dataset,
        batch_size=inference_args.per_device_batch_size,
        collate_fn=data_collator,
        num_workers=inference_args.dataloader_num_workers,
        pin_memory=True,
    )
    data_loader = accelerator.prepare(data_loader)
    total_inference_steps = len(data_loader)

    start_step = get_last_checkpoint_step(os.path.join(data_args.temporary_save_to_disk))
    accelerator.wait_for_everyone()
    if start_step > 0:
        logger.info(f"Resuming from step {start_step}")
        # efficiently skip the first n batches
        start_step += 1
        data_loader = skip_first_batches(data_loader, start_step) # TODO: do we really need this now?

    all_generated_labels = []
    accelerator.wait_for_everyone()
    if start_step < total_inference_steps:
        for i, batch in enumerate(tqdm(data_loader, disable=not accelerator.is_local_main_process)):
            cur_step = start_step + i
            generate_labels = apply_audio_decoder(batch)
            # generate_labels = accelerator.pad_across_processes(generate_labels, dim=0, pad_index=0)
            generate_labels = accelerator.gather_for_metrics(generate_labels)

            if accelerator.is_main_process:
                lab = generate_labels["labels"].cpu().to(torch.int16)

                all_generated_labels.extend(lab)

                if ((cur_step + 1) % data_args.save_data_steps == 0) or (
                    cur_step == total_inference_steps - 1
                ):
                    tmp_labels = Dataset.from_dict({"labels": all_generated_labels})
                    save_checkpoint(
                        os.path.join(data_args.temporary_save_to_disk), tmp_labels, cur_step, num_proc=num_workers
                    )
                    all_generated_labels = []

        accelerator.wait_for_everyone()

    if accelerator.is_main_process and len(all_generated_labels) > 0:
        tmp_labels = Dataset.from_dict({"labels": all_generated_labels})
        save_checkpoint(os.path.join(data_args.temporary_save_to_disk), tmp_labels, cur_step, num_proc=num_workers)
        all_generated_labels = []
    accelerator.wait_for_everyone()

    del all_generated_labels
    accelerator.wait_for_everyone()

    with accelerator.local_main_process_first():
        tmp_labels = load_all_checkpoints(os.path.join(data_args.temporary_save_to_disk)).select(
            range(len(dataset))
        )
        logger.info(f"Concatenating: {tmp_labels} with {dataset}")
        if data_args.remove_audio_column:
            dataset = concatenate_datasets([dataset.remove_columns(data_args.audio_column_name), tmp_labels], axis=1)
        else:
            dataset = concatenate_datasets([dataset, tmp_labels], axis=1)

    accelerator.free_memory()
    del generate_labels

    def postprocess_dataset(labels):
        output = {"labels": id2label[labels]}
        return output

    with accelerator.local_main_process_first():
        dataset = dataset.map(
            postprocess_dataset,
            num_proc=num_workers,
            input_columns=["labels"],
            desc="Postprocessing labeling",
        )

    if accelerator.is_main_process:
        dataset.save_to_disk(
            inference_args.output_dir,
            num_proc=data_args.preprocessing_num_workers,
        )
    accelerator.wait_for_everyone()
    logger.info(f"Dataset saved at {inference_args.output_dir}")



if __name__ == "__main__":
    main()
