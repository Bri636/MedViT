import pickle
from callbacks.knn_callback import make_confusion_matrix, make_tsne
from utils import BaseConfig
from medmnist_datasets.medmnist_dataset import make_dataloaders, _DATA_FLAGS
from VitLightning import LitMedViT
import itertools
from pathlib import Path
from typing import Tuple, Optional
from argparse import ArgumentParser
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import medmnist
from medmnist import INFO
# or MedViT_base, MedViT_large as required
from models.MedViT import MedViT_small
from tqdm import tqdm
import copy
from utils import GradCAM
import matplotlib
matplotlib.use('Agg')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_DIR = Path(__file__).resolve().parent
# fixed paths for stuff
KNN_WEIGHTS_PATH = f'{CURRENT_DIR}/weights/3_1.0_knn_train_weights'
# MODEL_CKPT_PATH = f'{CURRENT_DIR}/models_all_checkpoints/good_checkpoints/epoch10_batch1500.pth'
MODEL_CKPT_PATH = f'/homes/bhsu/2024_research/MedViT/models_all_checkpoints/final_checkpoints/epoch=10-step=5200.ckpt'
IMG_SAVE_PATH = f'{CURRENT_DIR}/images/'

def make_gradcam(model_gradcam: torch.nn.Module) -> GradCAM:
    """Make a GradCAM object for evaluating the model.

    If the provided model is a Lightning module wrapping the backbone in `model`,
    it uses that backbone to extract the first convolutional layer from the patch embedding.
    """
    # NOTE: this should be a lightning model
    model_gradcam.eval()
    backbone = model_gradcam.model if hasattr(
        model_gradcam, "model") else model_gradcam
    # Check for the expected attribute structure.
    if hasattr(backbone, "features") and hasattr(backbone.features[0], "patch_embed"):
        gradcam_layer = backbone.features[0].patch_embed.conv
    else:
        raise ValueError(
            "The model does not contain the expected 'features[0].patch_embed.conv' structure.")
    return GradCAM(model_gradcam, gradcam_layer)

def unnormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize a tensor image using mean=0.5 and std=0.5.
    Assumes img_tensor shape is (C, H, W).
    """
    return img_tensor * 0.5 + 0.5

def get_embeddings(dataloader: data.DataLoader, model: nn.Module, num_batches: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Extract embeddings and labels """
    embeddings_list = []
    labels_list = []
    model.eval()
    inputs, targets = next(iter(dataloader))
    B = inputs.size()[0]
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader,
                                    desc=f"Extracting Embeds Shape: {B}, Num Passes: {num_batches}",
                                    total=num_batches):
            inputs = inputs.to(DEVICE)
            emb = model(inputs)
            embeddings_list.append(emb.cpu())
            labels_list.append(targets.cpu())
    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).squeeze().numpy()
    return embeddings, labels

def visualize_gradcam(image_tensor: torch.Tensor,
                      heatmap: torch.Tensor,
                      estimated_label: int,
                      gold_label: int,
                    #   idx: int
                      ) -> Figure:
    """
    Displays a side-by-side figure with the original image and the Grad-CAM heatmap overlay.
    The figure title includes 'Estimated Label' (model prediction) and 'Gold Label' (true label).
    """
    image = unnormalize(image_tensor).permute(1, 2, 0).cpu().numpy()
    cam = heatmap.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(image)
    axes[1].imshow(cam, cmap='jet', alpha=0.5)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis('off')
    fig.suptitle(f"Estimated Label: {estimated_label}, Gold Label: {gold_label}")
    return fig
    
def generate_gradcam_plots(model: nn.Module,
                            dataloader: data.DataLoader,
                            batch_size: int,
                            knn: KNeighborsClassifier
                            ) -> list[Tuple[str, Figure]]:
    """ Generates Grad-CAM images of OCT using one batch from dataloder """
    gradcam = make_gradcam(model)

    inputs, targets = next(iter(dataloader))  # single batch
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        embeds: np.ndarray = model(inputs).cpu().numpy()

    min_bsz = min(batch_size, len(embeds)) # dataloader batch size might be  diff
    pred_labels = knn.predict(embeds)
    cams = gradcam(inputs)  # [B, 1, H, W]
    gradcam_iterable = zip(inputs, cams, pred_labels, targets)
    figs = []
    for idx, (image, cam, pred, gold) in tqdm(enumerate(gradcam_iterable),
                                        desc=f'Generating {min_bsz} Grad-CAM Images',
                                        total=min_bsz):
        gradcam_fig = visualize_gradcam(image, cam, pred, gold.item())
        title = f"{CURRENT_DIR}/final_images/GradCAM-Image-{idx}-Gold-{gold.item()}-Est-{pred}.png"
        figs.append((title, gradcam_fig))
    gradcam.remove_hooks()
    return figs

def parse_arguments():
    argparser = ArgumentParser()
    argparser.add_argument('--model_checkpoint_path',
                           type=str,
                           default=MODEL_CKPT_PATH)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--images_save_path',
                           type=str,
                           default=IMG_SAVE_PATH)
    argparser.add_argument('--train_ratio', type=float, default=0.01)
    argparser.add_argument('--knn_from_scratch', action='store_true')
    return argparser.parse_args()

def main():
    args = parse_arguments()
    dataloaders = make_dataloaders('octmnist', args.batch_size)
    train_dataloader = dataloaders['train']
    small_num_batches = int(len(train_dataloader) * args.train_ratio)
    train_dataloader = itertools.islice(train_dataloader, small_num_batches)

    test_dataloader = dataloaders['test']
    n_classes = dataloaders['n_classes']

    model = LitMedViT.load_from_checkpoint(
        checkpoint_path=args.model_checkpoint_path)
    model = model.to(DEVICE)
    model.eval()

    # KNN training
    knn_weights_file = str(KNN_WEIGHTS_PATH + '.pkl')
    if not os.path.exists(knn_weights_file) or args.knn_from_scratch:
        print(
            f'\nKNN Weights Not At: {knn_weights_file} or You Want to Train from Scratch, Training and Saving Weights...\n')
        train_embeddings, train_labels = get_embeddings(
            train_dataloader, model, small_num_batches)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_embeddings, train_labels)
        knnPickle = open(knn_weights_file, 'wb')
        pickle.dump(knn, knnPickle)
        knnPickle.close()
    else:
        print(
            f'\nWeights Found at {knn_weights_file}, Loading in from there...\n')
        knn = pickle.load(open(knn_weights_file, 'rb'))

    figs = generate_gradcam_plots(copy.deepcopy(model), test_dataloader, args.batch_size, knn)
    list(map(lambda x: x[1].savefig(x[0]), figs))

    X_test, y_test = get_embeddings(test_dataloader, model, len(test_dataloader))
    knn_pred_test = knn.predict(X_test)
    # plot the confusion matrix for the test set
    cm_fig = make_confusion_matrix(knn_pred_test, y_test)
    cm_fig.savefig('test_set_cm.png')
    tsne_fig = make_tsne(X_test, y_test, 'Test')
    tsne_fig.savefig('test_set_tsne.png')

    accuracy = accuracy_score(y_test, knn_pred_test)
    f1 = f1_score(y_test, knn_pred_test, average='weighted')
    prob_estimates = knn.predict_proba(X_test)
    if n_classes == 2:
        auc = roc_auc_score(y_test, prob_estimates[:, 1])
    else:
        auc = roc_auc_score(y_test, prob_estimates, multi_class='ovr')

    print("===================================")
    print("k-NN Evaluation on Test Set:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("===================================")
    print("\nClassification Report:")
    print(classification_report(y_test, knn_pred_test))

if __name__ == "__main__":
    main()
