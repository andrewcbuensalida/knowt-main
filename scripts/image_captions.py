from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor

from transformers import AutoModelForCausalLM
from evaluate import load
import torch


def plot_images(images, captions):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")


# sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
# sample_captions = [train_ds[i]["text"] for i in range(5)]
# plot_images(sample_images_to_visualize, sample_captions)


checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)


model = AutoModelForCausalLM.from_pretrained(checkpoint)


wer = load("wer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}
