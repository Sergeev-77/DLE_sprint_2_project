import gc
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.eval_lstm import calculate_rouge_lstm
from configs.config import Config

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory /1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            utilization = allocated / total_memory * 100
            print(f"\nGPU {i}===>:")
            print(f"total mem    : {total_memory:.2f} GB")
            print(f"!current mem : {allocated:.2f} GB ({utilization:.1f}%)")
            print(f"reserved mem : {reserved:.2f} GB")
            print(f"max per epoch: {max_allocated:.2f} GB")
            print(f"=====>GPU {i}]\n")
            torch.cuda.reset_peak_memory_stats()
    else:
        print("cuda не доступна")

def plot_metrics(metrics):
    epochs = range(1, len(metrics["train_losses"]) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    ax1.plot(epochs, metrics["train_losses"], "b-", label="train loss")
    ax1.plot(epochs, metrics["val_losses"], "r-", label="val loss")
    ax1.set_title("train and val losses")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(True)

    # ROUGE-1
    ax2.plot(epochs, metrics["val_rouge1"], "r-", label="val rouge1")
    ax2.set_title("rouge1 score")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("rouge1")
    ax2.legend()
    ax2.grid(True)

    # ROUGE-2
    ax3.plot(epochs, metrics["val_rouge2"], "r-", label="val rouge2")
    ax3.set_title("rouge2 score")
    ax3.set_xlabel("epochs")
    ax3.set_ylabel("rouge2")
    ax3.legend()
    ax3.grid(True)

    # Сравнение ROUGE метрик
    ax4.plot(epochs, metrics["val_rouge1"], "r-", label="val rouge1")
    ax4.plot(epochs, metrics["val_rouge2"], "r--", label="val rouge2")
    ax4.set_title("All rouge metrics")
    ax4.set_xlabel("epochs")
    ax4.set_ylabel("score")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def train_model(
    model,
    tokenizer,
    config: Config,
    train_dataloader,
    val_dataloader,
    val_sample_dataloader,
    criterion,
    optimizer,
):
    


    if config.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()        

    gc.collect()

    train_losses = []
    val_losses = []
    val_rouge1 = []
    val_rouge2 = []

    print(f"Модель на {next(model.parameters()).device}\n")

    for epoch in range(config.num_epochs):
        model.train()
        cur_train_loss = 0
        progress_bar = tqdm(
            train_dataloader, desc=f"train epoch {epoch+1}/{config.num_epochs}"
        )
        for inputs, targets, _ in progress_bar:
            inputs, targets = inputs.to(config.device), targets.to(
                config.device
            )
            optimizer.zero_grad()

            outputs, _ = model(inputs)

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            cur_train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = cur_train_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        if config.device == 'cuda':
            torch.cuda.empty_cache()        

        gc.collect()

        # Валидация
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for input_seq, target_seq, _ in val_dataloader:
                input_seq = input_seq.to(config.device)
                target_seq = target_seq.to(config.device)

                output, _ = model(input_seq)
                output = output.reshape(-1, output.size(-1))
                target_seq = target_seq.reshape(-1)

                loss = criterion(output, target_seq)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        val_r1, val_r2 = calculate_rouge_lstm(
            model, val_sample_dataloader, tokenizer, config, prefix="val_smpl"
        )

        val_rouge1.append(val_r1)
        val_rouge2.append(val_r2)

        print_gpu_memory()
        print(f"  train loss:      {avg_loss:.4f}, val loss: {avg_val_loss:.4f}")
        print(f"  val_smpl rouge1: {val_r1:.4f},   val_smpl rouge2: {val_r2:.4f}")
        print("\n", "-" * 20, f"end of epoch {epoch+1}/{10}", "-" * 20, "\n")

        if config.device == 'cuda':
            torch.cuda.empty_cache()        

        gc.collect()

    plot_metrics(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_rouge1": val_rouge1,
            "val_rouge2": val_rouge2,
        }
    )
