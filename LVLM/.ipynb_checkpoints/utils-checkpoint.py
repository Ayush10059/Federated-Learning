import math
import os
import json
import re
import warnings
import logging
import pandas as pd
from typing import List
from collections import OrderedDict
from typing import Callable, Dict, Tuple

import textwrap
from omegaconf import OmegaConf
from logging import WARNING, ERROR, LogRecord
import flwr as fl
from flwr_datasets import FederatedDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from flwr.common.logger import ConsoleHandler, console_handler, FLOWER_LOGGER, LOG_COLORS
from hydra import compose, initialize
from omegaconf import DictConfig
from datasets import load_dataset, Dataset, load_from_disk
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from sklearn.metrics import accuracy_score

from datasets import Dataset as HFDataset

from torch.utils.data import Dataset, DataLoader

import glob

import cv2

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from dataclasses import dataclass
from typing import Any, List

import logging

def format_string(msg, char_width: int=50) -> str:
    return textwrap.fill(msg, char_width, subsequent_indent="\t")


######### print Hydra config as yaml ##################
def print_config(config: DictConfig):
    print(OmegaConf.to_yaml(config))

########## console logger with less white spaces #############
FLOWER_LOGGER.removeHandler(console_handler) # remove default handler
class ConsoleHandlerV2(ConsoleHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def format(self, record: LogRecord) -> str:
        """Format function that adds colors to log level."""
        if self.json:
            log_fmt = "{lvl='%(levelname)s', time='%(asctime)s', msg='%(message)s'}"
        else:
            log_fmt = (
                f"{LOG_COLORS[record.levelname] if self.colored else ''}"
                f"%(levelname)s {'%(asctime)s' if self.timestamps else ''}"
                f"{LOG_COLORS['RESET'] if self.colored else ''}"
                f": %(message)s"
            )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure console logger
console_handlerv2 = ConsoleHandlerV2(
    timestamps=False,
    json=False,
    colored=True,
)
console_handlerv2.setLevel(logging.INFO)
FLOWER_LOGGER.addHandler(console_handlerv2)


########### to filter out all warnigns from HF coming from client side #########
backend_setup = {"logging_level": ERROR, "log_to_driver": False}

################################ configs components #############################


def get_config(config_name: str):
    with initialize(config_path="../LVLM/conf", version_base="1.1"):
        cfg = compose(config_name=config_name)

    return cfg


############################# visualize data partitions #######################
def visualize_partitions(fed_dataset: FederatedDataset):
    _ = fed_dataset.load_partition(0)
    num_partitions = fed_dataset.partitioners['train'].num_partitions
    
    plt.bar(range(num_partitions), [len(fed_dataset.load_partition(i)) for i in range(num_partitions)])
    plt.xticks(range(num_partitions))
    plt.xlabel("Partition ID")
    plt.ylabel("Number of examples")
    plt.title(f"IID partitioning into {num_partitions} partitions")


############################### Report communication costs #################

def compute_communication_costs(config, comm_bw_mbps: float = 20):
    model = get_model(config.model)

    trainable, all_parameters = model.get_nb_trainable_parameters()

    total_size = 4*all_parameters/(1024**2)
    trainable_size = 4*trainable/(1024**2)

    upload_time_total = total_size/(comm_bw_mbps/8)
    upload_time_finetune = trainable_size/(comm_bw_mbps/8)
    
    print(f"Full model:\n\t{all_parameters/1e6:.3f} M parameters\n\t{total_size:.2f} MB --> upload in {upload_time_total:.2f}s @ {comm_bw_mbps}Mbps")
    print(f"Finetuned model:\n\t{trainable/1e6:.3f} M parameters\n\t{trainable_size:.2f} MB --> upload in {upload_time_finetune:.2f}s @ {comm_bw_mbps}Mbps")
    # print(f"In a {comm_bw_mbps} Mbps channel --> {}")

    num_rounds = config.flower.num_rounds
    num_clients_per_round = int(config.flower.num_clients * config.flower.fraction_fit)
    print(f"Federated Learning setting: "
          f"\n\tNumber of rounds: {num_rounds}"
          f"\n\tNumber of clients per round: {num_clients_per_round}")
    
    print(f"-----------------------------------------------")
    print(f"Total Communication costs (Full model): {2*num_rounds*num_clients_per_round*total_size/1024:.1f} GB")
    print(f"Total Communication costs (Finetuning): {2*num_rounds*num_clients_per_round*trainable_size} MB")
    print(f"Communication savings: {all_parameters/trainable:.1f}x")


################################ model components #############################


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load the LLaVa-NeXT-Video model with quantization + LoRA if desired."""
    use_cuda = torch.cuda.is_available()
    # set quantization
    if model_cfg.quantization == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    elif model_cfg.quantization == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError("quantization must be 4 or 8")

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_cfg.name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    lora_config = LoraConfig(
        r=model_cfg.lora.r,
        lora_alpha=model_cfg.lora.alpha,
        target_modules=["q_proj","v_proj"],
        lora_dropout=model_cfg.lora.dropout,
        bias=model_cfg.lora.bias,
        task_type="CAUSAL_LM",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    if not use_cuda:
        model.enable_input_require_grads()
    if model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model


################################ dataset components #############################


def get_processor_and_data_collator(
    model_name: str, use_fast: bool = True, padding_side: str = "right"
):
    # load the LLaVA processor
    processor = LlavaNextVideoProcessor.from_pretrained(model_name, use_fast=True)
    # our custom video collator
    data_collator = VideoDataCollator(processor)
    # no extra formatting function
    return processor, data_collator


################################ client components #############################
def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)
    
class LlavaClient(fl.client.NumPyClient):
    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        dataset: Dataset,
        data_collator: Callable,
        save_path: str,
        client_id: int,
    ):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.dataset = dataset
        self.data_collator = data_collator
        self.save_path = save_path
        self.client_id = client_id

        self.model = None
        self.trainer = None

    def fit(self, parameters, config):
        if self.model is None:
            self.model = get_model(self.model_cfg)

        set_parameters(self.model, parameters)

        # Setup Trainer
        training_args = TrainingArguments(
            output_dir=f"{self.save_path}/client_{self.client_id}",
            per_device_train_batch_size=self.train_cfg.batch_size,
            gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
            learning_rate=self.train_cfg.learning_rate,
            num_train_epochs=self.train_cfg.num_train_epochs,
            max_steps=self.train_cfg.max_steps,
            logging_steps=self.train_cfg.logging_steps,
            save_steps=self.train_cfg.save_steps,
            fp16=True,
            remove_unused_columns=False,
            report_to="none",
            save_total_limit=self.train_cfg.save_total_limit,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=self.data_collator,
        )

        train_result = trainer.train()
        train_loss = train_result.training_loss if train_result.training_loss is not None else 0.0
        
        return self.get_parameters(), len(self.dataset), {"train_loss": train_loss}


    def get_parameters(self, config=None):
        if self.model is None:
            self.model = get_model(self.model_cfg)
        lora_state = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for val in lora_state.values()]

def gen_client_fn(
    data_dir: str,
    data_collator,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    save_path: str,
) -> Callable[[Context], LlavaClient]:
    def client_fn(context: Context) -> LlavaClient:
        torch.cuda.empty_cache()
        pid = int(context.node_config["partition-id"])

        client_path = os.path.join(data_dir, f"client_{pid}")
        video_paths = glob.glob(os.path.join(client_path, "*.avi"))

        # Collect (video_path, label) pairs
        video_samples = []
        for path in video_paths:
            fname = os.path.basename(path).lower()
            if "fight" in fname:
                label = 1
            else:
                label = 0
            video_samples.append({
                "video_path": path,
                "label": label,
            })

        # Use lazy dataset
        lazy_dataset = LazyVideoDataset(video_samples)

        return LlavaClient(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset=lazy_dataset,
            data_collator=data_collator,
            save_path=save_path,
            client_id=pid,
        )

    return client_fn
 

class LazyVideoDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        """
        samples: List[Dict] with 'video_path' and 'label'
        """
        self.samples = samples
        self.transform = transform  # e.g., frame sampling, resizing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        video_path = item['video_path']
        label = item['label']

        # Lazy-load a single frame (can be improved later)
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise ValueError(f"Could not read video: {video_path}")

        if self.transform:
            frame = self.transform(frame)

        return {
            "video": frame,
            "video_path": video_path,
            "label": label,
        }


@dataclass
class VideoDataCollator:
    """
    Builds LLaVA-style “chat” inputs from (video_path, label) examples.
    """
    processor: LlavaNextVideoProcessor
    instruction: str = "Analyze the video. Is this a fight scene? Answer yes or no."

    def __call__(self, batch: List[dict]) -> dict:
        conversations = []
        for ex in batch:
            video_path = ex["video_path"]
            label = ex["label"]
            answer = "yes" if label == 1 else "no"
            # user message: text + video
            user = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": "video", "path": video_path},
                ],
            }
            # assistant message: ground‑truth answer
            assistant = {
                "role": "assistant",
                "content":[{"type":"text","text":answer}],
            }
            conversations.append([user, assistant])

        # apply LLaVA processing: sample frames + tokenize
        model_inputs = self.processor.apply_chat_template(
            conversations,
            num_frames=24,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )

        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs

################################ server components #############################


# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
from sklearn.metrics import accuracy_score, roc_auc_score

def get_evaluate_fn(model_cfg, processor, save_path, data_dir):
    """
    Return an evaluation function for Flower that evaluates the global model
    using VideoEvaluator and returns metrics.
    """
    def evaluate(server_round: int, parameters, config):
        if server_round != config.get("num_rounds", -1):
            return 0.0, {}  # Skip evaluation on all rounds except final

        print(f"\nRunning evaluation on final global model at round {server_round}...")

        model = get_model(model_cfg)
        set_parameters(model, parameters)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        evaluator = VideoEvaluator(model, processor, model_cfg)
        results_df = evaluator.evaluate(data_dir)

        analyze_results(results_df)

        accuracy = accuracy_score(results_df["true"], results_df["pred"])
        return accuracy, {}

    return evaluate


# Get a function that will be used to construct the config that the client's
# fit() method will receive
def get_on_fit_config(num_rounds):
    def fit_config_fn(server_round: int):
        fit_config = {
            "current_round": server_round,
            "num_rounds": num_rounds,
        }
        return fit_config
    return fit_config_fn



def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


################################# evaluation ##########################
class VideoEvaluator:
    def __init__(self, model, processor, model_cfg):
        self.model = model
        self.processor = processor
        self.model_cfg = model_cfg
        self.device = model.device
        
    def predict(self, video_path):
        """Make prediction for a single video"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": "Analyze the video. Is this a fight scene? Answer with only yes or no"},
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            num_frames=self.model_cfg.num_frames,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, torch.float16)
        
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        prediction = self.processor.batch_decode(
            outputs, 
            skip_special_tokens=True
        )[0].strip().lower().split()[-1].replace(".", "")

        return 1 if "yes" in prediction else 0
    
    def evaluate(self, dataset):
        """Full evaluation on dataset"""
        results = []
        for video_path, true_label in tqdm(
            zip(dataset["videos"], dataset["labels"]),
            total=len(dataset),
            desc="Evaluating"
        ):
            print(video_path)
            try:
                pred_label = self.predict(video_path)
                results.append({
                    "video": video_path,
                    "true": true_label,
                    "pred": pred_label
                })
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                
        return pd.DataFrame(results)

def analyze_results(results_df):
    """Comprehensive results analysis"""
    # Metrics
    accuracy = accuracy_score(results_df["true"], results_df["pred"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        results_df["true"], results_df["pred"], average="binary"
    )
    roc_auc = roc_auc_score(results_df["true"], results_df["pred"])
    
    # Confusion Matrix
    cm = confusion_matrix(results_df["true"], results_df["pred"])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Fight", "Fight"],
                yticklabels=["Non-Fight", "Fight"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(results_df["true"], results_df["pred"])
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(results_df["true"], results_df["pred"]))