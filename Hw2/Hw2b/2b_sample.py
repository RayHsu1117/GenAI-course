import transformers as T
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from ignite.metrics import Rouge
import re
device = "cuda" if torch.cuda.is_available() else "cpu"




def get_tensor(sample):
    # 將模型的輸入和ground truth打包成Tensor
    model_inputs = t5_tokenizer.batch_encode_plus([each["concepts"] for each in sample], padding=True, truncation=True, return_tensors="pt")
    model_outputs = t5_tokenizer.batch_encode_plus([each["targets"] for each in sample], padding=True, truncation=True, return_tensors="pt")
    return model_inputs["input_ids"].to(device), model_outputs["input_ids"].to(device)

class CommonGenDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        data_df = load_dataset("allenai/common_gen", split=split, cache_dir="./cache/").to_pandas().groupby("concept_set_idx")
        self.data = []
        for each in data_df:
            targets = "/ ".join([s+"." if not s.endswith(".") else s for s in each[1].target.to_list()])
            concepts = ", ".join(each[1].concepts.to_list()[0])
            self.data.append({"concepts": concepts, "targets": targets})

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

data_sample = CommonGenDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")


lr = 1e-5
epochs = 1
optimizer = AdamW(t5_model.parameters(), lr = 1e-5)
train_batch_size = 8
validation_batch_size = 8
common_gen_train = DataLoader(CommonGenDataset(split="train"), collate_fn=get_tensor, batch_size=train_batch_size, shuffle=True)
common_gen_validation = DataLoader(CommonGenDataset(split="validation"), collate_fn=get_tensor, batch_size=validation_batch_size, shuffle=False)
rouge = Rouge(variants=["L", 2], multiref="best")


def evaluate(model):
    pbar = tqdm(common_gen_validation)
    pbar.set_description(f"Evaluating")

    for inputs, targets in pbar:
        output = [re.split(r"[/]", each.replace("<pad>", "")) for each in t5_tokenizer.batch_decode(model.generate(inputs, max_length=50))]
        targets = [re.split(r"[/]", each.replace("<pad>", "")) for each in t5_tokenizer.batch_decode(targets)]
        for i in range(len(output)):
            sentences = [s.replace('.', ' .').split() for s in output[i]]
            ground_thruths = [t.replace('.', ' .').split() for t in targets[i]]
            for s in sentences:
                rouge.update(([s], [ground_thruths]))
    return rouge.compute()


for ep in range(epochs):
    pbar = tqdm(common_gen_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    for inputs, targets in pbar:
        optimizer.zero_grad()
        loss = t5_model(input_ids=inputs, labels=targets).loss
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss = loss.item())
    torch.save(t5_model, f'./saved_models/ep{ep}.mod')
    print(f"Rouge-2 score on epoch {ep}:", evaluate(t5_model))