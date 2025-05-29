import torch
import normflows as nf
import pickle

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from library.cnf_functions import load_and_prepare_data, split_data, convert_to_tensors, plot_initial_distribution, plot_training_and_validation_loss, plot_trained_distribution, evaluate_model

# LaTeX style setup for matplotlib
plt.rc('font', family='serif', serif=['Computer Modern'])
plt.rc('text', usetex=True)

# SETUP #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

forward_csv = './data/forward_data.csv'

data_to_learn, conditioning_data, normalisation_params = \
    load_and_prepare_data(forward_csv)
train_data, valid_data, test_data, train_cond, valid_cond, test_cond = \
    split_data(data_to_learn, conditioning_data)
train_data, valid_data, test_data, train_cond, valid_cond, test_cond = \
    convert_to_tensors(device, train_data, valid_data, test_data,
                       train_cond, valid_cond, test_cond)

plot_initial_distribution(data_to_learn, conditioning_data)

# DEFINE MODEL #
K = 8
latent_size = train_data.shape[1]
context_size = train_cond.shape[1]
hidden_units = 128
hidden_layers = 4

flows = []
for i in range(K):
    flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
    flows += [nf.flows.LULinearPermute(latent_size)]

q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
model = nf.ConditionalNormalizingFlow(q0, flows, None)
model.to(device)

# TRAINING #
batch_size = 128
epochs = 80
patience = 80  # Early stopping patience
epsilon = 1e-10  # small constant to avoid log(0)

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.96)

train_loss_hist = []
val_loss_hist = []
lrs = []

best_val_loss = float('inf')
early_stopping_rounds = 0

# creating DataLoader for training and validation data
train_dataset = TensorDataset(train_data, train_cond)
valid_dataset = TensorDataset(valid_data, valid_cond)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    num_batches = 0

    for x, context in train_loader:
        optimiser.zero_grad()
        # compute the KLD loss
        log_prob = model.log_prob(x, context)
        kld_loss = -torch.mean(log_prob + epsilon)  # add epsilon inside the log

        if not (torch.isnan(kld_loss) | torch.isinf(kld_loss)):
            kld_loss.backward()
            optimiser.step()
            epoch_train_loss += kld_loss.item()
            num_batches += 1

    scheduler.step()
    train_loss_hist.append(epoch_train_loss / num_batches)

    lrs.append(optimiser.param_groups[0]['lr'])

    # validation step
    if epoch % 1 == 0:
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x_val, context_val in valid_loader:
                val_log_prob = model.log_prob(x_val, context_val)
                val_loss = -torch.mean(val_log_prob + epsilon).item()  # add epsilon inside the log
                epoch_val_loss += val_loss
        val_loss_hist.append(epoch_val_loss / len(valid_loader))

        print(f'Epoch {epoch}: Training Loss {epoch_train_loss / num_batches}')
        print(f'Epoch {epoch}: Validation Loss {epoch_val_loss / len(valid_loader)}')

    # Early stopping check
    if val_loss_hist[-1] < best_val_loss:
        best_val_loss = val_loss_hist[-1]
        early_stopping_rounds = 0  # Reset early stopping counter
    else:
        early_stopping_rounds += 1

    if early_stopping_rounds >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# SAVE MODEL #
model_save_path = './models/CNF.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

with open('./models/normalisation_params.pkl', 'wb') as file:
    pickle.dump(normalisation_params, file)

# EVALUATION AND PLOTS #
evaluate_model(model, test_data, test_cond)   # quantitative
plot_training_and_validation_loss(train_loss_hist, val_loss_hist, lrs)   # loss
plot_trained_distribution(test_data, test_cond, model)    # trained distribution