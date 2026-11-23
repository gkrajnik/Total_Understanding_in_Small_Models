import torch as th
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from alg_zoo.tasks import *

def hidden_state_evolution(x, model):
    x = x.to(th.float32)
    model.eval()
    with th.no_grad():
        out = model.rnn(x.unsqueeze(dim=-1))
    fig, axes = plt.subplots(figsize = (6,5))
    sns.heatmap(out[0][0], ax= axes, cmap = 'viridis')
    axes.set_xlabel('Hidden dimensions')
    axes.set_ylabel('Hidden states')
    plt.show()

def print_params(model):
    for name, param in model.named_parameters():
        print(name, param.size())

def plot_weights(model):
    fig, axes = plt.subplots(1, 3, figsize = (13,5), gridspec_kw={'width_ratios':[4, 20, 20]})
    sns.heatmap(model.rnn._flat_weights[0].detach(), ax= axes[0], cmap = 'viridis')
    W_hh = model.rnn._flat_weights[1].detach()
    sns.heatmap(W_hh, ax= axes[1], cmap = 'viridis')
    for i in range(W_hh.shape[0]):
        axes[1].add_patch(patches.Rectangle((i, i), 1, 1, fill=False, edgecolor='white', lw=1.5))
    sns.heatmap(model.linear.weight.detach(), ax= axes[2], cmap = 'viridis')
    pos = axes[1].get_position()
    axes[1].set_position([pos.x0 + 0.005, pos.y0, pos.width, pos.height])
    axes[0].set_title('W_hi')
    axes[1].set_title('W_hh')
    axes[2].set_title('W_oh')
    plt.show()



def plot_logits(x, model):
    x = x.to(th.float32)
    model.eval()
    with th.no_grad():
        out = model.rnn(x.unsqueeze(dim=-1))
        logits = model.linear(out[1]).squeeze(0)
        pred = model(x).squeeze(0).argmax().int()
    fig, axes = plt.subplots(figsize = (10,2))
    sns.heatmap(logits, ax= axes, cmap='RdBu_r', center=True)
    axes.set_xlabel('Hidden dimensions')
    axes.set_ylabel('Logits')
    plt.subplots_adjust(bottom= 0.3)
    fig.text(0, 0.1, f"x = {x[0].int().tolist()}") 
    fig.text(0, 0.04, f"actual index = {task_2nd_argmax(x.squeeze(0))}, predicted index ={pred}") 
    plt.show()

def get_rescaled_weights(model):
    """
    Returns rescaled weight matrices for an equivalent model where W_hi has ±1 entries.
    Works because ReLU is positively homogeneous: relu(D⁻¹ z) = D⁻¹ relu(z) for positive D.

    D = diag(|W_hi|)
    W_hi' = D⁻¹ W_hi        (entries become ±1)
    W_hh' = D⁻¹ W_hh D
    W_oh' = W_oh D
    """
    W_hi = model.rnn._flat_weights[0].detach()  # (hidden_size, 1)
    W_hh = model.rnn._flat_weights[1].detach()  # (hidden_size, hidden_size)
    W_oh = model.linear.weight.detach()          # (output_size, hidden_size)

    d = W_hi[:, 0].abs()  # (hidden_size,)

    W_hi_r = W_hi / d[:, None]
    W_hh_r = W_hh / d[:, None] * d[None, :]
    W_oh_r = W_oh * d[None, :]

    return W_hi_r, W_hh_r, W_oh_r


def plot_weights_rescaled(model):
    W_hi_r, W_hh_r, W_oh_r = get_rescaled_weights(model)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), gridspec_kw={'width_ratios': [4, 20, 20]})
    sns.heatmap(W_hi_r.detach(), ax=axes[0], cmap='viridis', vmin=-1, vmax=1)
    sns.heatmap(W_hh_r.detach(), ax=axes[1], cmap='viridis')
    for i in range(W_hh_r.shape[0]):
        axes[1].add_patch(patches.Rectangle((i, i), 1, 1, fill=False, edgecolor='white', lw=1.5))
    sns.heatmap(W_oh_r.detach(), ax=axes[2], cmap='viridis')
    pos = axes[1].get_position()
    axes[1].set_position([pos.x0 + 0.005, pos.y0, pos.width, pos.height])
    axes[0].set_title("W_hi'")
    axes[1].set_title("W_hh'")
    axes[2].set_title("W_oh'")
    plt.show()


if __name__ == "__main__":
    from alg_zoo import example_2nd_argmax
    model = example_2nd_argmax()
    x= th.tensor([[2, 4, 5, 5, 8, 21, 33, 9, 1, 0]], dtype= th.float32)
    plot_logits(x, model)



